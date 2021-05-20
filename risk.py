import time
import warnings

import tqdm
import numba
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


MAX_ATTACKING_DICE = 3
MAX_DEFENDING_DICE = 2
MIN_DICE_ROLL = 1
MAX_DICE_ROLL = 6

warnings.simplefilter("ignore", lineno=2119)

@numba.jit
def roll(attack_dice, defend_dice):
    """Simulate a single dice roll in a battle.
    
    :param attack_dice: The number of attacking dice
    :type attackers: int
    :param defend_dice: The number of defending dice
    :type attackers: int
    :returns: A tuple of (attacker losses, defender losses)
    :rtype: tuple[int]
    """
    n_compare = min(attack_dice, defend_dice)
    attackers = np.sort(np.random.randint(MIN_DICE_ROLL,MAX_DICE_ROLL+1, attack_dice))[-n_compare:]
    defenders = np.sort(np.random.randint(MIN_DICE_ROLL,MAX_DICE_ROLL+1, defend_dice))[-n_compare:]
    attack_losses = np.sum(attackers <= defenders)
    return attack_losses, n_compare-attack_losses

@numba.jit
def battle(attackers=1, defenders=1):
    """Simulate one battle between an attacker and a defender.
    
    :param attackers: Number of attacking armies, defaults to 1
    :type attackers: number, optional
    :param defenders: Number of defending armies, defaults to 1
    :type defenders: number, optional
    :returns: A tuple of the final number of (attackers, defenders). One will
        always be zero.
    :rtype: tuple[int]
    """
    if attackers == 0 or defenders == 0:
        return attackers, defenders
    else:
        attack_loss, defense_loss = roll(
            min(attackers, MAX_ATTACKING_DICE), 
            min(defenders, MAX_DEFENDING_DICE)
        )
        return battle(attackers - attack_loss, defenders - defense_loss)

@numba.jit
def battle_sequence(attackers, defense_chain):
    """Simulate an attacking horde attacking through a chain of defenders.
    
    :param attackers: The number of attacking armies
    :type attackers: int
    :param defense_chain: The chain of defending armies
    :type defense_chain: list[int]
    :returns: The number of attackers left after each attack in the chain.
    :rtype: ndarray[int]
    """
    attack_chain = np.zeros(len(defense_chain))
    for i, defenders in enumerate(defense_chain):
        attackers, _ = battle(attackers, defenders)
        if attackers == 0:
            break
        attackers -= 1
        attack_chain[i] = attackers

    return attack_chain


def battle_sequences(attackers, defense_chain, n=100_000):
    """Simulate an attacking horde attacking through a chain of defenders.

    Runs many (n) simulations for calculating statistics.
    
    :param attackers: The number of attacking armies
    :type attackers: int
    :param defense_chain: The chain of defending armies
    :type defense_chain: list[int]
    :param n: The number of simulations to run, defaults to 100_000.
    :type n: int
    :returns: The number of attackers left after each attack in the chain.
        One row per simulation, one column per attack.
    :rtype: ndarray[int]
    """
    attack_chains = np.empty(shape=(n, len(defense_chain)), dtype=int)
    for i in tqdm.tqdm(range(n), desc="Simulating", total=n):
        attack_chains[i,:] = battle_sequence(attackers, defense_chain)
    return attack_chains

def battle_sequence_report(attackers, defense_chain, n=100_000):
    """Visualize the expected outcome of a given battle sequence using 
    simulation.
    
    :param attackers: The number of attacking armies
    :type attackers: int
    :param defense_chain: The chain of defending armies
    :type defense_chain: list[int]
    :param n: The number of simulations to run, defaults to 100_000.
    :type n: int
    :returns: The number of attackers left after each attack in the chain.
        One row per simulation, one column per attack.
    :rtype: ndarray[int]
    """
    start = time.perf_counter()
    attack_chains = battle_sequences(attackers, defense_chain, n)
    end = time.perf_counter()
    
    n_steps = attack_chains.shape[-1]
    n_cols = min(4, n_steps-1)
    n_rows = 1 + int(np.ceil((n_steps-1)/n_cols)) if n_cols else 1

    fig = plt.figure(figsize=(10,(n_rows+2)*2), facecolor="white")
    bins = np.arange(0, attack_chains.max()+2) - 0.5
    ax = None
    for i in range(n_steps-1):
        ax = plt.subplot(n_rows, n_cols, i+1, sharey=ax)
        sns.histplot(attack_chains[:,i], bins=bins, ax=ax, color=f"C{i}", stat="density")
        ax.set_xlabel("Attackers remaining")
        ax.set_ylabel("Probability")
        ax.set_title(f"{'After: ' if i==0 else ''}{i+1} attack{'s' if i>0 else ''}")
    
    ax = plt.subplot(n_rows, 1, n_rows)
    sns.histplot(attack_chains[:,-1], bins=bins, ax=ax, color=f"C{n_steps-1}", stat="density")
    ax.set_xlabel("Attackers remaining")
    ax.set_ylabel("Probability")
    ax.set_title(f"After all attacks")

    prob_success = (attack_chains[:,-1]>0).mean()
    mean_left = attack_chains[:,-1].mean()
    chain_text = ', '.join([str(int(i)) for i in defense_chain])
    message = (
        f"{attackers} armies attacking a chain of: {chain_text}\n"
        f"Simulated {n:,} trials in {end-start:.2f} seconds\n"
        f"Probability of success: {prob_success:.2%}\n"
        f"Average number of attackers remaining: {mean_left:.2f}\n\n"
        "Expected distribution of remaining attackers after each attack:"
    )
    fig.suptitle(message, fontsize=16)
    
    plt.tight_layout()
    plt.show()

    return attack_chains