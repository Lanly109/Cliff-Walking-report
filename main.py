import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

epos = 5000

n = 3
m = 12

class Cliff_Walking():
    def __init__(self, n = 3, m = 12):
        self.n = n
        self.m = m
        self.reward = np.zeros((n, m))
        self.reward[:, :] = -1
        self.reward[2, 1:m - 1] = -100
        self.reward[n - 1, m - 1] = -1
        self.action = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def ax2sq(self, status):
        return status // self.m, status % self.m

    def sqaction(self, action):
        return self.action[action]

    def sq2ax(self, x, y):
        return x * self.m + y

    def step(self, status, action):
        x, y = self.ax2sq(status)
        dx, dy = self.sqaction(action)
        if x + dx >= 0 and x + dx < self.n:
            x = x + dx
        if y + dy >= 0 and y + dy < self.m:
            y = y + dy
        reward = self.reward[x, y]
        if x == self.n - 1 and y > 0 and y < self.m - 1:
            x = 0
            y = 0
        return self.sq2ax(x, y), reward, x == self.n - 1 and y == self.m - 1

class Sarsa():
    def __init__(self, statue_n = 36, action_n = 4, gamma = 0.9, alpha = 0.3, epsilon = 0.1):
        self.action_n = action_n
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((statue_n, action_n))
        self.next_action = None

    def choose_action(self, status):
        if self.next_action is not None:
            return self.next_action
        else:
            return self.choosse_action(status)

    def choosse_action(self, status):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            return self.choose_best(status)

    def choose_best(self, status):
        return self.Q[status, :].argmax();

    def learn(self, status, action, reward, next_status, done):
        old_Q = self.Q[status, action]
        if done:
            next_Q = reward
        else:
            self.next_action = self.choosse_action(next_status)
            next_Q = reward + self.gamma * self.Q[next_status, self.next_action]
        self.Q[status, action] = old_Q + self.alpha * (next_Q - old_Q)

    def save(self):
        npy_file = './sarsa_table.txt'
        np.savetxt(npy_file, self.Q)
        print(npy_file + ' saved.')


class Qlearning():
    def __init__(self, statue_n = 36, action_n = 4, gamma = 0.9, alpha = 0.3, epsilon = 0.1):
        self.action_n = action_n
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((statue_n, action_n))

    def choose_action(self, status):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            return self.choose_best(status)

    def choose_best(self, status):
        return self.Q[status, :].argmax();

    def learn(self, status, action, reward, next_status, done):
        old_Q = self.Q[status, action]
        if done:
            next_Q = reward
        else:
            next_Q = reward + self.gamma * self.Q[next_status, self.choose_best(next_status)]
        self.Q[status, action] = old_Q + self.alpha * (next_Q - old_Q)

    def save(self):
        npy_file = './Qlearning_table.txt'
        np.savetxt(npy_file, self.Q)
        print(npy_file + ' saved.')


def run(learner):
    env = Cliff_Walking()
    pos = 24
    done = False
    totle_reward = 0
    totle_step = 0
    while True:
        action = learner.choose_action(pos)
        next_pos, reward, done = env.step(pos, action)
        totle_reward += reward
        totle_step += 1
        learner.learn(pos, action, reward, next_pos, done)
        if done:
            break
        pos = next_pos
    return totle_reward, totle_step


def test(learner, name):
    env = Cliff_Walking(n, m)
    pos = 24
    done = False
    totle_reward = 0
    tu = np.zeros((n, m))
    while True:
        if done:
            break
        action = learner.choose_best(pos)
        tu[env.ax2sq(pos)] = action + 1
        next_pos, reward, done = env.step(pos, action)
        pos = next_pos
        totle_reward += reward
    print(f"-------------{name}-------------")
    for i in range(n):
        for j in range(m):
            if i == n - 1 and j == 0:
                print('S ', end='')
            elif i == n - 1 and j == m - 1:
                print('T ', end='')
            elif tu[i][j] == 0:
                print('. ', end='')
            elif tu[i][j] == 1:
                print('➡️ ', end='')
            elif tu[i][j] == 2:
                print('⬅️ ', end='')
            elif tu[i][j] == 3:
                print('⬇️ ', end='')
            elif tu[i][j] == 4:
                print('⬆️ ', end='')
        print('')
    print('')


def train_and_test(Learn, name):
    areward = []
    jun = []
    cnt = 0
    learner = Learn(n * m)
    stepnum = []
    juns = []
    for i in range(epos):
        reward, step = run(learner)
        if cnt < 700:
            areward.append(reward)
            stepnum.append(step)
        else:
            areward[cnt % 700] = reward
            stepnum[cnt % 700] = step
        jun.append(np.mean(areward))
        juns.append(np.mean(stepnum))
        cnt += 1

    test(learner, name)
    return jun, learner.Q, juns


sarsa, sarsa_Q, sarsa_step = train_and_test(Sarsa, "Sarsa")
qlearning, qlearning_Q, qlearning_step = train_and_test(Qlearning, "Qlearning")

f, (ax1,ax2) = plt.subplots(nrows=2)

sq = []
for i in sarsa_Q:
    sq.append(np.max(i[:]))
qq = []
for i in qlearning_Q:
    qq.append(np.max(i[:]))

sq = np.array(sq).reshape(n, m)
sq = pd.DataFrame(sq, columns=np.arange(1,m+1), index=np.arange(1,n+1))
qq = np.array(qq).reshape(n, m)
qq = pd.DataFrame(qq, columns=np.arange(1,m+1), index=np.arange(1,n+1))

sns.heatmap(sq, ax = ax1, fmt = 'd', cmap='rainbow')
sns.heatmap(qq, ax = ax2, fmt = 'd', cmap='rainbow')

x = range(epos)
plt.figure(2)
plt.plot(x, sarsa)
plt.plot(x, qlearning)
plt.ylim(-1000, 20)

plt.figure(3)
plt.plot(x, sarsa_step)
plt.plot(x, qlearning_step)
plt.show()



