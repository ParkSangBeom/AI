import tensorflow as tf
import numpy as np
import gym
from collections import deque
from datetime import datetime

env = gym.make("Breakout-v0");

SAVE_EPISODE = tf.Variable(0, name = "save_episode", dtype = tf.int32)

HEIGHT_SIZE = 80
WIDTH_SIZE = 80
COLOR_SIZE = 1
INPUT_SIZE = HEIGHT_SIZE * WIDTH_SIZE * COLOR_SIZE
OUTPUT_SIZE = 2

EPISODE_MAX = 100000
DISCOUNT = 0.8
EPSILON = 1e-10

CHECK_POINT_PATH = "./CheckPoint/"
TENSORBOARD_PATH = "./Tensorboard/"
IS_RENDER = True
SAVE_INTERVAL = 50

class PolicyGradientNetwork:
    _LEARNING_RATE = 0.0003
    _HIDDEN_SIZE1 = 100
    _HIDDEN_SIZE2 = 100

    def __init__(self, sess, height_size, width_size, color_size, output_size, name = "NoneNetwork"):
        self.sess = sess
        self.height_size = height_size
        self.width_size = width_size
        self.color_size = color_size        
        self.output_size = output_size
        self.name = name

        self._BuildNetwork()
        self._SettingTensorboard()

    def _BuildNetwork(self):
        with tf.variable_scope(self.name):
            self.observation = tf.placeholder(dtype = tf.float32, shape = [None, self.height_size, self.width_size, self.color_size])
            self.action = tf.placeholder(dtype = tf.float32, shape = [None, self.output_size])
            self.reward = tf.placeholder(dtype = tf.float32, shape = [None])
            self.ori_reward = tf.placeholder(dtype = tf.float32, shape = [None])

            with tf.name_scope("Conv1"):
                F1 = tf.Variable(tf.random_normal([3, 3, 1, 4], stddev=0.01))
                L1 = tf.nn.conv2d(self.observation, F1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope("Conv2"):
                F2 = tf.Variable(tf.random_normal([3, 3, 4, 8], stddev=0.01))
                L2 = tf.nn.conv2d(L1, F2, strides=[1, 1, 1, 1], padding='SAME')
                L2 = tf.nn.relu(L2)
                self.L2_flat = tf.reshape(L2, [-1, L2.shape[1] * L2.shape[2] * L2.shape[3]])
                #print(self.L2_flat.shape)

        input_layer = tf.layers.dense(self.L2_flat, units=self._HIDDEN_SIZE1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.layers.dense(input_layer, units=self.output_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.get_action = tf.reshape(tf.multinomial(self.logits, 1), [])
        self.loss = tf.losses.softmax_cross_entropy(self.action, logits=self.logits, weights=self.reward)
        self.train = tf.train.AdamOptimizer(self._LEARNING_RATE).minimize(self.loss)
        self.saver = tf.train.Saver()

        tf.summary.histogram("Logits", self.logits)
        tf.summary.scalar("Loss_Value", self.loss)
        tf.summary.scalar("Reward_Value", tf.reduce_sum(self.ori_reward))

    def _SettingTensorboard(self):
        now = datetime.now()
        dir = TENSORBOARD_PATH + self.name + "/" + \
        str(now.year) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(dir)
        self.writer.add_graph(self.sess.graph)

    def GetAction(self, obs):
        action = self.sess.run(self.get_action, feed_dict={self.observation : np.reshape(obs, [1, self.height_size, self.width_size, self.color_size])})
        return action

def Train(PG, obs, act, rew, episode, ori_reward):
    _, summary = PG.sess.run([PG.train, PG.merged_summary], feed_dict={PG.observation : obs, PG.action : act, PG.reward : rew, PG.ori_reward : ori_reward})
    PG.writer.add_summary(summary, global_step = episode)

def OneHot(value):
    zero = np.zeros(OUTPUT_SIZE, dtype = np.int)
    zero[value] = 1
    return  zero

def ProcessSaverAndRestore(PG):
    if not os.path.exists(CHECK_POINT_PATH):
        os.makedirs(CHECK_POINT_PATH)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_PATH)

    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(PG.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        except:
            print("Error on loading old network weights")
    else:
        print("Could not find old network weights")

    return saver

def PreprocessImage(image):
    image = image[35:195]
    image = image[::2, ::2]
    image[image == 144] = 0
    image[image == 109] = 0
    image[image != 0] = 1
    image = np.mean(image, axis = 2)
    image = np.reshape(image, [HEIGHT_SIZE, WIDTH_SIZE, COLOR_SIZE])
    return image.astype(np.float)

def ProcessReward(cur_lives, life, reward):    
    lives = life['ale.lives']
    if cur_lives == -1:
        cur_lives = lives
        env.step(1)
        env.step(1)
        env.step(1)

    if cur_lives > lives:
        cur_lives = lives
        reward = -1.0
        env.step(1)
        env.step(1)
        env.step(1)

    if reward > 0.:
        reward = 1.0

    return cur_lives, reward

def DiscountRewards(reward_memory):
    v_memory = np.vstack(reward_memory)
    discounted = np.zeros_like(v_memory, dtype=np.float32)
    add_value = 0
    length = len(reward_memory)

    for i in reversed(range(length)):
        if v_memory[i] < 0:
            add_value = 0
        add_value = v_memory[i] + (DISCOUNT * add_value)
        discounted[i] = add_value

    discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + EPSILON)
    return discounted

def main():
    with tf.Session() as sess:
        PG = PolicyGradientNetwork(sess, HEIGHT_SIZE, WIDTH_SIZE, COLOR_SIZE, OUTPUT_SIZE, "SpaceInvaders")
        PG.sess.run(tf.global_variables_initializer())
        saver = ProcessSaverAndRestore(PG)
        episode = PG.sess.run(SAVE_EPISODE) + 1

        while(episode < EPISODE_MAX):

            obs_memory = deque()
            action_memory = deque()
            reward_memory = deque()
            show_memory = deque();

            cur_lives = -1;
            step = 1
            done = False

            pre_obs = env.reset()
            cur_obs = pre_obs = PreprocessImage(pre_obs)

            while not done:

                if IS_RENDER == True:
                    env.render()

                obs_delta = cur_obs - pre_obs
                pre_obs = cur_obs

                #[0. Left] [1. CenterF] [2. Right] [3. Center] [4. RightF] [5. LeftF]
                action = PG.GetAction(obs_delta)
                one_hot = OneHot(action)

                cur_obs, reward, done, life = env.step(action + 2)
                cur_obs = PreprocessImage(cur_obs)

                cur_lives, reward = ProcessReward(cur_lives, life, reward)

                obs_memory.append(obs_delta)
                action_memory.append(one_hot)
                reward_memory.append(reward)
                if reward > 0:
                    show_memory.append(reward)                
                if done:
                    rewards = DiscountRewards(reward_memory)
                    Train(PG, np.stack(obs_memory, axis=0), np.stack(action_memory, axis =0), np.reshape(rewards, [-1]), episode, show_memory)
                    print("[EPISODE :", episode, "]  Step :", step, "  Reward :", np.sum(show_memory))

                step += 1

            if episode % SAVE_INTERVAL == 0:
                PG.sess.run(SAVE_EPISODE.assign(episode))
                saver.save(PG.sess, CHECK_POINT_PATH + "/model", global_step = episode)
                print("=====[Save :", episode, "]=====")

            episode += 1

if __name__ == "__main__":
    main()