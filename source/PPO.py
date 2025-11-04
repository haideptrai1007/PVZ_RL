import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np

class GridNet(nn.Module):
    def __init__(self, input_channels=13):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(input_channels, 32, 
                                    kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.upsample(x))  
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.adaptive_pool(x)  
        x = self.conv2(x) 
        x = x.view(x.size(0), -1) 
        return x

class Context(nn.Module):
    def __init__(self, input_channels=10):
        super().__init__()
        self.L1 = nn.Linear(input_channels, 32)
        self.L2 = nn.Linear(32, 64)
        self.L3 = nn.Linear(64, 128)
    
    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        
        return x

class ActorCritic(nn.Module):
    def __init__(self, grid_h=5, grid_w=9, n_plants=9, n_zoms=5):
        super(ActorCritic, self).__init__()

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_plants = n_plants
        self.n_zoms = n_zoms


        self.Net = GridNet()
        self.Context = Context()
        self.ActionOneHot = Context(n_plants)

        self.actor_plant = nn.Sequential(
            nn.Linear(1152, 512), # 1024 (Net) + 128(Context)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.n_plants)
        )

        self.actor_grid = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.grid_h * self.grid_w)
        )

        self.critic = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, plant_action=None):
        grid_obs, context_obs = state

        gridFtrs = self.Net(grid_obs)
        contextFtrs = self.Context(context_obs)
        ftrs = torch.cat([gridFtrs, contextFtrs], dim=-1)

        plant_logits = self.actor_plant(ftrs)
        plant_probs = F.softmax(plant_logits, dim=-1)

        if plant_action is None:
            plant_onehot = plant_probs
        else:
            plant_onehot = F.one_hot(plant_action, num_classes=self.n_plants).float()

        actionFtrs = self.ActionOneHot(plant_onehot)
        actorGridFtrs = torch.cat([gridFtrs, actionFtrs], dim=-1) 

        grid_logits = self.actor_grid(actorGridFtrs)
        grid_probs = F.softmax(grid_logits, dim=-1)

        value = self.critic(ftrs)

        return plant_probs, grid_probs, value

    def get_action(self, state, gridMask=None):
        _, context_obs = state

        plant_probs, _, value = self.forward(state)
        actionMask = context_obs[:, :-1]
        
        plant_probs *= actionMask
        plant_probs /= (plant_probs.sum(dim=-1, keepdim=True) + 1e-6)

        plant_dist = Categorical(plant_probs)
        plant_action = plant_dist.sample()

        plant_log_prob = plant_dist.log_prob(plant_action)

        _, grid_probs, _ = self.forward(state, plant_action)

        if gridMask is not None:
            grid_probs *= gridMask
            grid_probs /= (grid_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        grid_dist = Categorical(grid_probs)
        grid_action = grid_dist.sample()
        grid_log_prob = grid_dist.log_prob(grid_action)

        return plant_action, grid_action, plant_log_prob, grid_log_prob, value

    def evaluate_actions(self, state, plant_action, grid_action):
            plant_probs, grid_probs, value = self.forward(state, plant_action)

            plant_dist = Categorical(plant_probs)
            plant_log_prob = plant_dist.log_prob(plant_action)
            plant_entropy = plant_dist.entropy()

            grid_dist = Categorical(grid_probs)
            grid_log_prob = grid_dist.log_prob(grid_action)
            grid_entropy = grid_dist.entropy()

            entropy = plant_entropy + grid_entropy

            return plant_log_prob, grid_log_prob, entropy, value
    
class RolloutBuffer:
    def __init__(self):
        self.grid_states = []
        self.context_states = []
        self.plant_actions = []
        self.grid_actions = []
        self.plant_log_probs = []
        self.grid_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, grid_state, context_state, plant_action, grid_action, plant_log_prob, grid_log_prob, reward, value, done):
        self.grid_states.append(grid_state)
        self.context_states.append(context_state)
        self.plant_actions.append(plant_action)
        self.grid_actions.append(grid_action)
        self.plant_log_probs.append(plant_log_prob)
        self.grid_log_probs.append(grid_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self):
        return {
            'grid_states': np.array(self.grid_states),
            "context_states": np.array(self.context_states),
            'plant_actions': np.array(self.plant_actions),
            'grid_actions': np.array(self.grid_actions),
            'plant_log_probs': np.array(self.plant_log_probs),
            'grid_log_probs': np.array(self.grid_log_probs),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'dones': np.array(self.dones)
        }
    
    def clear(self):
        self.grid_states.clear()
        self.context_states.clear
        self.plant_actions.clear()
        self.grid_actions.clear()
        self.plant_log_probs.clear()
        self.grid_log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)
    
class PPOAgent:
    def __init__(self,
                grid_h=5, 
                grid_w=9, 
                n_plants=9, 
                n_zoms=5, 
                lr=3e-4, 
                gamma=0.9, 
                gae_lamb=0.95, 
                eps=0.2, 
                value_coef=1, 
                entropy_coef=0.1,
                max_grad_norm=0.5,
                ppo_epochs=4,
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

        self.policy = ActorCritic(grid_h, grid_w, n_plants, n_zoms).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def select_action(self, state, gridMask=None):
        grid_state, context_state = state
        grid_state = torch.FloatTensor(grid_state).to(self.device)
        context_state = torch.FloatTensor(context_state).to(self.device)
        with torch.no_grad():
            plant_action, grid_action, plant_log_prob, grid_log_prob, value = self.policy.get_action([grid_state.unsqueeze(0), context_state.unsqueeze(0)], gridMask)

        self.buffer.add(
            grid_state.cpu().numpy(),
            context_state.cpu().numpy(),
            plant_action.cpu().numpy()[0],
            grid_action.cpu().numpy()[0],
            plant_log_prob.cpu().numpy()[0],
            grid_log_prob.cpu().numpy()[0],
            0,
            value.cpu().numpy()[0][0],
            False

        )

        return plant_action.item(), grid_action.item()
    
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

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        grid_states = data["grid_states"]
        context_states = data["context_states"]

        grid_states = torch.FloatTensor(grid_states).to(self.device)
        context_states = torch.FloatTensor(context_states).to(self.device)
        plant_actions = torch.LongTensor(data["plant_actions"]).to(self.device)
        grid_actions = torch.LongTensor(data["grid_actions"]).to(self.device)
        old_plant_log_probs = torch.LongTensor(data["plant_log_probs"]).to(self.device)
        old_grid_log_probs = torch.LongTensor(data["grid_log_probs"]).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for i in range(self.ppo_epochs):
            indices = np.arange(len(data["grid_states"]))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                batch_states = [grid_states[batch_indices], context_states[batch_indices]]

                batch_plant_actions = plant_actions[batch_indices]
                batch_grid_actions = grid_actions[batch_indices]
                batch_old_plant_log_probs = old_plant_log_probs[batch_indices]
                bactch_old_grid_log_probs = old_grid_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                new_plant_log_probs, new_grid_log_probs, entropy, values = self.policy.evaluate_actions(batch_states, batch_plant_actions, batch_grid_actions)

                old_log_probs = batch_old_plant_log_probs + bactch_old_grid_log_probs
                new_log_probs = new_plant_log_probs + new_grid_log_probs

                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

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

