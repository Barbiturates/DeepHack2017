#!/usr/bin/env python3

import argparse
import datetime
import json
import logging
import os
import pickle
import random
import time

import gym
import numpy as np
import pandas as pd
import pytz
import tqdm

import agents

from utils.memory import Memory

FRAME_SKIPPING = 4

gym.undo_logger_setup()
log = logging.getLogger(name=__name__)


def load_gym_env(game_name='Skiing-v0'):
    log.debug('Loading {} environment'.format(game_name))
    env = gym.make('Skiing-v0')
    env.reset()
    return env


def main(agent_name,
         render=True,
         upload=False,
         monitor=False,
         slow=0.0,
         n_episodes=None,
         seed=None,
         agent_args='{}'):
    if not monitor and upload:
        raise ValueError('Cannot upload without monitoring!')

    # load the gym
    env = load_gym_env()

    # Initialize memory
    memory = Memory()

    # loaf the agent
    agent = agents.Random()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        env._seed(seed)

    eastern = pytz.timezone('Europe/Moscow')
    timestamp = datetime.datetime.now(eastern).strftime(
        '%Y-%m-%d__%H-%M-%S')

    local_dir = os.path.dirname(__file__)
    results_dir = os.path.join(local_dir, 'results', agent_name, timestamp)
    os.makedirs(results_dir)

    if monitor:
        # don't let the monitor log at anything below info
        monitor_level = max(logging.INFO, log.getEffectiveLevel())
        logging.getLogger(
            'gym.monitoring.video_recorder').setLevel(monitor_level)

        # start monitoring results
        env.monitor.start(results_dir)

    episode_count = env.spec.trials if n_episodes is None else n_episodes
    max_steps = env.spec.timestep_limit

    # track total rewards
    total_rewards = []
    reward_log = []

    try:
        # use a progress bar unless debug logging
        with tqdm.tqdm(total=episode_count * max_steps,
                       disable=log.getEffectiveLevel() < logging.INFO) as pbar:
            # for each episode
            for episode in range(episode_count):
                if hasattr(agent, 'episode_number'):
                    episode = agent.episode_number

                total_reward = 0.0

                image = env.reset()

                reward = 0

                for iteration in range(max_steps):
                    # update progress bar
                    pbar.update(n=1)

                    if render:
                        env.render()

                    # ask the agent what to do next
                    action = agent.act(image, centiseconds=-reward)

                    # take the action and get the new state and reward
                    new_image, reward, done, _ = env.step(action)
                    total_reward += reward

                    # Update memory
                    memory.add(image, action, reward, new_image, done)

                    # feed back to the agent on every FRAME_SKIPPING frame
                    if iteration % FRAME_SKIPPING == 0:
                        batch = memory.get_batch()
                        agent.react(
                            batch,
                            centiseconds=((-reward) % 10) + 1
                        )

                    if done:
                        # calculate components of reward
                        pos_reward = int(-reward)
                        goal_reward = pos_reward - (pos_reward % 500)
                        slaloms_missed = goal_reward / 500
                        if slaloms_missed == 0 and total_reward == -30000:
                            slaloms_missed = 20

                        pbar.update(max_steps - iteration - 1)
                        break
                    else:
                        # update the old state
                        image = new_image

                    # slow down the simulation if desired
                    if slow > 0.0:
                        time.sleep(slow)

                # timeout the sim
                if iteration == max_steps:
                    msg = 'Episode {} timed out after {} steps'.format(
                        episode, max_steps
                    )
                    log.debug(msg)

                msg = (
                    'Episode {} ({} steps): {}/{}, (Sloth: {}, Slaloms Missed: {})'.format(
                        episode,
                        iteration,
                        int(total_reward),
                        int(total_reward + 15000),
                        int(total_reward + goal_reward),
                        slaloms_missed
                    )
                )
                log.debug(msg)

                total_rewards.append(total_reward)

                reward_log.append({
                    'episode': episode,
                    'reward': total_reward,
                    'sloth': int(total_reward + goal_reward),
                    'missed': slaloms_missed
                })

                if episode % 100 == 0 and episode != 0:
                    log.debug('100 episode average reward was {}'.format(np.mean(total_rewards[-100:])))
                    # save the model
                    agent_path = os.path.join(
                        results_dir, 'agent_{}.pkl'.format(episode)
                    )
                    with open(agent_path, 'wb') as fout:
                        pickle.dump(agent, fout)

        log.debug('Last 100 episode average reward was {}'.format(
            np.mean(total_rewards[-100:])))
        log.debug('Best {}-episode average reward was {}'.format(
            episode_count, np.mean(total_rewards)
        ))

    finally:
        if monitor:
            # Dump result info to disk
            env.monitor.close()

        # debugging output
        if hasattr(agent, 'data') and agent.data is not None:
            df = pd.DataFrame(agent.data)
            df.to_csv(os.path.join(results_dir, 'data.csv'))

        # rewards output
        df = pd.DataFrame(reward_log)
        df.to_csv(os.path.join(results_dir, 'rewards.csv'))

        log.info(
            'Average reward of last 100 episodes: {}'.format(
                df.reward.values[-100:].mean())
        )
        log.info(
            "Average cost of elapsed time over last 100 episodes: {}".format(
                df.sloth.values[-100:].mean())
        )
        log.info(
            'Average number of slaloms missed over last 100 episodes: {}'.format(
                df.missed.values[-100:].mean())
        )

        with open(os.path.join(results_dir, 'agent_args.json'), 'w') as fout:
            fout.write(agent_args)

    if upload:
        # Upload to the scoreboard.
        log.info('Uploading results from {}'.format(results_dir))
        gym.upload(results_dir)


if __name__ == '__main__':
    main('')
