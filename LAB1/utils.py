import numpy as np
import matplotlib.pyplot as plt
import os

def izhikevich(a, b, c, d, current, u, condition, w=-1, tau=0.25, max_t=200, T1=-1, 
               equalize=True, current_type="step", current_eq=None, beta=0, feature=None,
               u1=0.04, u2=5, u3=140):
    """
    Izhikevich model of spiking neurons.

    Args:
        a, b, c, d: parameters of the model.
        current: input current
        u: membrane potential initial value
        condition: is a lambda function that takes t and T1 and decide I value
        w: recovery variable initial value
        tau: time step of the simulation
        max_t: maximum time of the simulation
        T1: time to apply the input current
        equalize: if true the plot cuts all spikes to max of 30
        current_type: if "step" the current is a step or pulse function, if "linear" the current is a linear increasing function
        beta: default current value for linear increasing current
        u1: first parameter of potential in Izhikevich model  (0.04)
        u2: second parameter of potential in Izhikevich model (5)
        u3: third parameter of potential in Izhikevich model  (140)
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
        elif current_type == "thr_var":
            I = input_current(t, feature)

        # leap-frog integration on Izhikevich model
        u = u + tau * (u1 * (u ** 2) + u2 * u + u3 - w + I )
        if feature != "(R)":
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
    if feature == "(O)":
        I = 1 if (10 < t < 15) or (80 < t < 85) else (-6 if 70 < t < 75 else 0)
    elif feature == "(R)":
        I = t / 25 if t < 200 else (0 if t < 300 or t >= 312.5 else (t - 300) / 12.5 * 4)
    return I

    

def plot_and_save(x, y, xlabel, ylabel, title, folder):
    """Genera e salva un grafico con i dati forniti."""
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
    """Genera e salva i grafici del potenziale di membrana e del ritratto di fase."""
    plot_and_save(tspan, pot, "Time (t)", "Membrane Potential (u)", 
                  f"{title} - Membrane Potential Plot", "potential_plots")

    plot_and_save(pot, rec, "Membrane Potential (u)", "Recovery variable (w)", 
                  f"{title} - Phase Portrait", "phase_portraits")