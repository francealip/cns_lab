import numpy as np
import matplotlib.pyplot as plt
import os

# utility functions for lab 1 mandatory part

def izhikevich(a, b, c, d, current, u, condition, w=-1, tau=0.25, max_t=200, T1=-1, 
               equalize=True, current_type="step", current_eq=None, beta=0,
               u2=5, u3=140):
    """
    Simulates the Izhikevich model of spiking neurons.

    Args:
        a, b, c, d (float): parameters of the Izhikevich model.
        current (float): Input current.
        u (float): Initial value of the membrane potential.
        condition (function): Lambda function that takes t and T1 and decides the value of I.
        w (float, optional): Initial value of the recovery variable. Defaults to -1 if w = b * u.
        tau (float, optional): Time step of the simulation. Defaults to 0.25.
        max_t (float, optional): Maximum time of the simulation. Defaults to 200.
        T1 (float, optional): Time to apply the input current. Defaults to -1.
        equalize (bool, optional): If True, the plot cuts all spikes to a maximum of 30. Defaults to True.
        current_type (str, optional): Type of current ("step" for step function or "pulses", "linear" for linear increasing function). Defaults to "step".
        current_eq (function, optional): Function defining the current if current_type is "linear". Defaults to None.
        beta (float, optional): Default current value. Defaults to 0.
        u2 (float, optional): Second parameter of potential in Izhikevich model. Defaults to 5.
        u3 (float, optional): Third parameter of potential in Izhikevich model. Defaults to 140.

    Returns:
        tuple: Membrane potential, recovery variable, and time span arrays.
    """
    
    # variables initializations
    w = b * u if w == -1 else w
    tspan = np.arange(0, max_t+tau, tau) 
    T1 = tspan[-1] / 10 if T1 == -1 else T1
    
    # potential and recovery variable values trough time 
    potential = []
    recovery = []

    for t in tspan: 
        # input current value
        if current_type == "step":
            I = current if condition(t, T1) else beta  
        elif current_type == "linear":
            I = current_eq(t, T1) if t > T1 else beta
        else:
            I = input_current(t, current_type)

        # leap-frog integration on Izhikevich model
        u = u + tau * (0.04 * (u ** 2) + u2 * u + u3 - w + I )
        if current_type != "acc_(R)":
            w = w + tau * a * (b * u - w)
        else:
            w = w + tau * a * (b * (u + 65))
        # reset condition
        if not(equalize):
            potential.append(u) 
        if u >= 30:
            if equalize:
                potential.append(30)
            u = c 
            w = w + d
        elif equalize:
            potential.append(u)
        recovery.append(w)
        
    return potential, recovery, tspan


def input_current (t, feature):
    """
    Returns the value of the input current for the Izhikevich model in cases of (O) and (R) features.
    
    Args:
        t (float): Time.
        feature (str): Feature for input current variation.
        
    Returns:
        float: Value of the input current.
    """
    if "(O)" in feature:
        I = 1 if (10 < t < 15) or (80 < t < 85) else (-6 if 70 < t < 75 else 0)
    elif "(R)" in feature:
        I = t / 25 if t < 200 else (0 if t < 300 or t >= 312.5 else (t - 300) / 12.5 * 4)
    return I


def plot_and_save(x, y, xlabel, ylabel, title, folder):
    """
    Generates and saves a plot.
    """
    
    size = (7, 4)
    name = title.replace(" ", "_")
    filepath = f"plots/{folder}/{name}"

    plt.figure(figsize=size)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, 'b', linewidth=0.5)

    if os.path.isfile(filepath):
        os.remove(filepath)
    plt.savefig(filepath)
    plt.show()

def plot_and_save_imgs(pot, rec, tspan, title):
    """
    Generates and saves the membrane potential and phase portrait plots.
    """
    
    plot_and_save(tspan, pot, "Time (t)", "Membrane Potential (u)", 
                  f"{title} - Membrane Potential Plot", "potential_plots")

    plot_and_save(pot, rec, "Membrane Potential (u)", "Recovery variable (w)", 
                  f"{title} - Phase Portrait", "phase_portraits")
    
    

# utility functions for lab 1 bonus track

def izhikevich_bonus(a, b, c, d, current, u, condition, w=-1, tau=0.25, max_t=200, T1=-1, 
                     beta=0, u2=5, u3=140):
    """
    Simulates the Izhikevich model of spiking neurons.

    Args:
        a, b, c, d (float): parameters of the Izhikevich model.
        current (float): Input current.
        u (float): Initial value of the membrane potential.
        condition (function): Lambda function that takes t and T1 and decides the value of I.
        w (float, optional): Initial value of the recovery variable. Defaults to -1 if w = b * u.
        tau (float, optional): Time step of the simulation. Defaults to 0.25.
        max_t (float, optional): Maximum time of the simulation. Defaults to 200.
        T1 (float, optional): Time to apply the input current. Defaults to -1.
        beta (float, optional): Default current value. Defaults to 0.
        u2 (float, optional): Second parameter of potential in Izhikevich model. Defaults to 5.
        u3 (float, optional): Third parameter of potential in Izhikevich model. Defaults to 140.

    Returns:
        tuple: Membrane potential, recovery variable, time span arrays and input current values.
    """
    
    # variables initializations
    w = b * u if w == -1 else w
    tspan = np.arange(0, max_t+tau, tau) 
    T1 = tspan[-1] / 10 if T1 == -1 else T1
    
    # potential and recovery variable values trough time 
    potential = []
    recovery = []
    in_curr = []    # input current values in the simulation
    in_plot = []    # input current values for plotting

    for t in tspan: 
        I = current if condition(t, T1) else beta  
        in_curr.append(I)
        if t % 10 in [0,1,2,3,4,5,6,7,8,9]:
            in_plot.append(I)
        
        # leap-frog integration on Izhikevich model
        u = u + tau * (0.04 * (u ** 2) + u2 * u + u3 - w + I )
        w = w + tau * a * (b * u - w)

        if u >= 30:
            potential.append(30)
            u = c 
            w = w + d
        else:
            potential.append(u)
        recovery.append(w)
        
    return potential, recovery, tspan, in_curr, in_plot

def plot_extra(pot, rec, tspan, in_plot, offset, title):
    size = (10, 4)  
    fig, axes = plt.subplots(1, 2, figsize=size)  

    # Membrane potential over time
    axes[0].set_title(title + " - Membrane potential over time")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Membrane potential (u)")
    axes[0].plot(tspan, pot, 'b', linewidth=0.5)
    axes[0].plot([x - offset for x in in_plot], 'r', linewidth=0.5)

    # Phase-plane plot 
    axes[1].set_title(title + " - Phase-plane plot")
    axes[1].set_xlabel("membrane potential (u)")
    axes[1].set_ylabel("recovery variable (w)")
    axes[1].plot(pot, rec)

    plt.tight_layout()  
    
    filepath = "plots/bonus_track/" + title.replace("-","").replace(" ","_")
    if os.path.isfile(filepath):
        os.remove(filepath)
    plt.savefig(filepath)
    
    
    plt.show()