import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import pickle


class GridNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)           
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc = nn.Linear(32*2*4, out_channels)   
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)               
        x = x.view(x.size(0), -1)      
        x = self.fc(x)              
        return x

class ActorCritic(nn.Module):
    def __init__(self, grid_h, grid_w, n_plants):
        super(ActorCritic, self).__init__()

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_plants = n_plants

        self.gridObs = GridNet(2, 32)

        self.critic = nn.Sequential(
            nn.Linear(32 + n_plants + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.plantAct = nn.Sequential(
            nn.Linear(32 + n_plants + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_plants)
        )

        self.laneAct = nn.Sequential(
            nn.Linear(32 + n_plants, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),            
            nn.Linear(64, grid_h)
        )

        self.posAct = nn.Sequential(
            nn.Linear(32 + grid_h + n_plants, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),    
            nn.Linear(64, grid_w) 
        )

    def forward(self, gridState, contextState, plantAction=None, laneAction=None):
        gridFtrs = self.gridObs(gridState)
        obs = torch.cat([gridFtrs, contextState], dim=-1)

        value = self.critic(obs)
        plantLogits = self.plantAct(obs)
        plantProbs = F.softmax(plantLogits, dim=-1)

        if plantAction is not None:
            plantOneHot = F.one_hot(plantAction, num_classes=self.n_plants).float()
        else:
            plantOneHot = plantProbs

        laneObs = torch.cat([gridFtrs, plantOneHot], dim=-1)
        laneLogits = self.laneAct(laneObs)
        laneProbs = F.softmax(laneLogits, dim=-1)

        if laneAction is not None:
            laneOneHot = F.one_hot(laneAction, num_classes=self.grid_h).float()
        else:
            laneOneHot = laneProbs

        posObs = torch.cat([laneObs, laneOneHot], dim=-1)
        posLogits = self.posAct(posObs)
        posProbs = F.softmax(posLogits, dim=-1)

        return plantProbs, laneProbs, posProbs, value
    
    def get_action(self, gridState, contextState, gridMask):
        # Plants Forward
        plantProbs, _, _, value = self.forward(gridState, contextState)

        plantMask = contextState[:, :-1]
        plantProbs *= plantMask
        plantProbs /= (plantProbs.sum(dim=-1, keepdim=True) + 1e-6)

        plantDist = Categorical(plantProbs)
        plantAction = plantDist.sample()
        plantLogProb = plantDist.log_prob(plantAction)

        # Lanes Forward
        _, laneProbs, _, _ = self.forward(gridState, contextState, plantAction)

        laneMask = gridMask.any(dim=1, keepdim=True).float().reshape(self.grid_h)

        laneProbs *= laneMask
        laneProbs /= (laneProbs.sum(dim=-1, keepdim=True) + 1e-6)

        laneDist = Categorical(laneProbs)
        laneAction = laneDist.sample()
        laneLogProb = laneDist.log_prob(laneAction)

        # Positions Forward
        _, _, posProbs, _ = self.forward(gridState, contextState, plantAction, laneAction) 

        posMask = gridMask[laneAction]

        posProbs *= posMask
        posProbs /= (posProbs.sum(dim=-1, keepdim=True) + 1e-6)

        posDist = Categorical(posProbs)
        posAction = posDist.sample()
        posLogProb = posDist.log_prob(posAction)

        return plantAction, laneAction, posAction, plantLogProb, laneLogProb, posLogProb, value
    
    def inference_action(self, gridState, contextState, gridMask):
        # Plants Forward
        plantProbs, _, _, value = self.forward(gridState, contextState)
        plantMask = contextState[:, :-1]

        plantProbs *= plantMask
        plantProbs /= (plantProbs.sum(dim=-1, keepdim=True) + 1e-6)
        plantAction = plantProbs.argmax(dim=-1)

        # Lanes Forward
        _, laneProbs, _, _ = self.forward(gridState, contextState, plantAction)
        laneMask = gridMask.any(dim=1, keepdim=True).float().reshape(self.grid_h)

        laneProbs *= laneMask
        laneProbs /= (laneProbs.sum(dim=-1, keepdim=True) + 1e-6)
        laneAction = laneProbs.argmax(dim=-1)
        # Positions Forward
        _, _, posProbs, _ = self.forward(gridState, contextState, plantAction, laneAction) 
        posMask = gridMask[laneAction]

        posProbs *= posMask
        posProbs /= (posProbs.sum(dim=-1, keepdim=True) + 1e-6)
        posAction = posProbs.argmax(dim=-1)

        return plantAction, laneAction, posAction, value


    def evaluate_actions(self, gridState, contextState, plantAction, laneAction, posAction):
        plantProbs, laneProbs, posProbs, values = self.forward(gridState, contextState, plantAction, laneAction)

        # Eval Plant
        plantDist = Categorical(plantProbs)
        plantLogProb = plantDist.log_prob(plantAction)
        plantEntropy = plantDist.entropy()

        # Eval Lane
        laneDist = Categorical(laneProbs)
        laneLogProb = laneDist.log_prob(laneAction)
        laneEntropy = laneDist.entropy()

        # Eval Pos
        posDist = Categorical(posProbs)
        posLogProb = posDist.log_prob(posAction)
        posEntropy = posDist.entropy()

        entropy = plantEntropy + laneEntropy + posEntropy

        return plantLogProb, laneLogProb, posLogProb, entropy, values    
    
class RolloutBuffer:
    def __init__(self):
        self.grid_states = []
        self.context_states = []

        self.plant_actions = []
        self.grid_actions = []
        self.pos_actions = []
        
        self.plant_log_probs = []
        self.grid_log_probs = []
        self.pos_log_probs = []
        
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, grid_state, context_state, plant_action, grid_action, pos_action, plant_log_prob, grid_log_prob, pos_log_probs, reward, value, done):
        self.grid_states.append(grid_state)
        self.context_states.append(context_state)

        self.plant_actions.append(plant_action)
        self.grid_actions.append(grid_action)
        self.pos_actions.append(pos_action)

        self.plant_log_probs.append(plant_log_prob)
        self.grid_log_probs.append(grid_log_prob)
        self.pos_log_probs.append(pos_log_probs)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self):
        return {
            'grid_states': np.array(self.grid_states),
            "context_states": np.array(self.context_states),
            'plant_actions': np.array(self.plant_actions),
            'grid_actions': np.array(self.grid_actions),
            'pos_actions': np.array(self.pos_actions),
            'plant_log_probs': np.array(self.plant_log_probs),
            'grid_log_probs': np.array(self.grid_log_probs),
            'pos_log_probs': np.array(self.pos_log_probs),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'dones': np.array(self.dones)
        }
        
    def clear(self):
        self.grid_states.clear()
        self.context_states.clear
        self.plant_actions.clear()
        self.grid_actions.clear()
        self.pos_actions.clear()
        self.plant_log_probs.clear()
        self.grid_log_probs.clear()
        self.pos_log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, filename="memory.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename="memory.pkl"):
        with open(filename, "rb") as f:
            self.__dict__.update(pickle.load(f))

    def __len__(self):
        return len(self.rewards)
    
class PPOAgent:
    def __init__(self,
                lr=3e-4, 
                gamma=0.99, 
                gae_lamb=0.99, 
                eps=0.2, 
                value_coef=1, 
                entropy_coef=0.1,
                max_grad_norm=0.5,
                ppo_epochs=2,
                mini_batch_size=256):
        

        self.gamma = gamma
        self.gae_lambda = gae_lamb
        self.eps = eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(5, 4, 5).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def select_action(self, grid_state, context_state, gridMask, inference_mode=False):
        grid_state = torch.FloatTensor(grid_state).to(self.device)
        context_state = torch.FloatTensor(context_state).to(self.device)

        if inference_mode:
            with torch.no_grad():
                plantAction, laneAction, posAction, value = self.policy.inference_action(grid_state.unsqueeze(0), context_state.unsqueeze(0), gridMask)
                return plantAction.item(), laneAction.item(), posAction.item()
  
        with torch.no_grad():
            plantAction, laneAction, posAction, plantLogProb, laneLogProb, posLogProb, value = self.policy.get_action(grid_state.unsqueeze(0), context_state.unsqueeze(0), gridMask)
  
        self.buffer.add(
            grid_state.cpu().numpy(),
            context_state.cpu().numpy(),

            plantAction.cpu().numpy()[0],
            laneAction.cpu().numpy()[0],
            posAction.cpu().numpy()[0],

            plantLogProb.cpu().numpy()[0],
            laneLogProb.cpu().numpy()[0],
            posLogProb.cpu().numpy()[0],

            0,
            value.cpu().numpy()[0][0],
            False

        )
        return plantAction.item(), laneAction.item(), posAction.item()
    
    def store_reward_and_done(self, reward, done):
        if len(self.buffer) > 0:
            self.buffer.rewards[-1] = reward
            self.buffer.dones[-1] = done

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[i+1]

            delta = rewards[i] + self.gamma * next_value_t * (1 - dones[i]) - values[i]

            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * last_gae
            advantages[i] = last_gae

        returns = advantages + values
        
        return advantages, returns
    
    def update(self, next_state=None):
        data = self.buffer.get()

        if next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, next_value = self.policy(next_state_tensor)
                next_value = next_value.cpu().numpy()[0][0]
        else:
            next_value = 0
        
        advantages, returns = self.compute_gae(data["rewards"], data["values"], data["dones"], next_value)

        grid_states = data["grid_states"]
        context_states = data["context_states"]

        grid_states = torch.FloatTensor(grid_states).to(self.device)
        context_states = torch.FloatTensor(context_states).to(self.device)

        plant_actions = torch.LongTensor(data["plant_actions"]).to(self.device)
        grid_actions = torch.LongTensor(data["grid_actions"]).to(self.device)
        pos_actions = torch.LongTensor(data["pos_actions"]).to(self.device)

        old_plant_log_probs = torch.LongTensor(data["plant_log_probs"]).to(self.device)
        old_grid_log_probs = torch.LongTensor(data["grid_log_probs"]).to(self.device)
        old_pos_log_probs = torch.LongTensor(data["pos_log_probs"]).to(self.device)

        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.arange(len(data["grid_states"]))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                # Rollout
                # States
                batch_grid_states = grid_states[batch_indices]
                batch_context_states = context_states[batch_indices]

                # Actions
                batch_plant_actions = plant_actions[batch_indices]
                batch_grid_actions = grid_actions[batch_indices]
                batch_pos_actions = pos_actions[batch_indices]
                
                # Action Probs
                batch_old_plant_log_probs = old_plant_log_probs[batch_indices]
                batch_old_grid_log_probs = old_grid_log_probs[batch_indices]
                batch_old_pos_log_probs = old_pos_log_probs[batch_indices]

                # Adv + Returns
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                newPlantLogProb, newLaneLogProb, newPosLogProb, entropy, values = self.policy.evaluate_actions(batch_grid_states, batch_context_states, batch_plant_actions, batch_grid_actions, batch_pos_actions)
                
                
                # Ratio
                ratio_plant = torch.exp(newPlantLogProb - batch_old_plant_log_probs)
                ratio_lane = torch.exp(newLaneLogProb - batch_old_grid_log_probs)
                ratio_pos = torch.exp(newPosLogProb - batch_old_pos_log_probs)


                # PPO Plant
                surr1_plant = ratio_plant * batch_advantages
                surr2_plant = torch.clamp(ratio_plant, 1 - self.eps, 1 + self.eps) * batch_advantages
                policy_loss_plant = -torch.min(surr1_plant, surr2_plant).mean()

                # PPO Lane
                surr1_lane = ratio_lane * batch_advantages
                surr2_lane = torch.clamp(ratio_lane, 1 - self.eps, 1 + self.eps) * batch_advantages
                policy_loss_lane = -torch.min(surr1_lane, surr2_lane).mean()

                # PPO Position
                surr1_pos = ratio_pos * batch_advantages
                surr2_pos = torch.clamp(ratio_pos, 1 - self.eps, 1 + self.eps) * batch_advantages
                policy_loss_pos = -torch.min(surr1_pos, surr2_pos).mean()


                policy_loss = policy_loss_plant + policy_loss_lane + policy_loss_pos
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

                num_updates += 1

        self.buffer.clear()

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def save(self, path):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

