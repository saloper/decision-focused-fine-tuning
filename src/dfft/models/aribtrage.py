import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

class BatteryArbitrage():
    """
    Class to hold a linear program for battery arbitrage. Parameters are:
    -T: task horizon
    -p_max: maximum power transfer
    -e_max: maximum energy storage
    -eta_c: charge efficiency
    -eta_d: dis-charge efficiency 
    """
    def __init__(self, T=24, p_max = 1, e_max = 4, eta_c = 0.95, eta_d = 0.95):
        #Define decision variables
        self.c = cp.Variable(T, nonneg = True) #charge
        self.d = cp.Variable(T, nonneg = True) #discharge
        self.s = cp.Variable(T + 1, nonneg = True)  #state of batttey with extra so we contrain no ending energy

        #Build Constraints
        self.constraints = [
            self.s[0] == 0, #Start with no power
            self.s[T] == 0, #end with no power
            self.c <= p_max, #max throughput
            self.d <= p_max, #max throughput
            self.s <= e_max #max charge
        ]

        #Link states of charge
        for t in range(T):
            self.constraints.append(self.s[t+1] == self.s[t] + (self.c[t] * eta_c) - (self.d[t] / eta_d))

        #Define prices
        self.prices = cp.Parameter(T)

        #Set Objective
        self.objective = cp.Maximize(self.prices @ self.d - self.prices @ self.c)

        #Define problem
        self.problem = cp.Problem(self.objective, self.constraints)

    def solve(self, prices):
        
        self.prices.value = prices
       
        self.problem.solve()

        return {
            "profit": self.problem.value,
            "charge_schedule": self.c.value,
            "discharge_schedule": self.d.value,
            "state_of_charge": self.s.value
        }
    
    def plot_solution(self):
        """
        Visualizes the optimal dispatch schedule, prices, and state of charge.
        Must be called after solve().
        """
        if self.prices.value is None or self.c.value is None:
            print("Error: You must call solve() before plotting.")
            return

        prices = self.prices.value
        charge = self.c.value
        discharge = self.d.value
        soc = self.s.value
        T = len(prices)

        # Create a 2-panel plot sharing the x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # ==========================================
        # TOP PLOT: Prices & Dispatch Actions
        # ==========================================
        # Plot the LMP prices on the primary y-axis
        ax1.plot(range(T), prices, label='LMP Price', color='black', marker='.', linewidth=2)
        ax1.set_ylabel('Price ($/MWh)', color='black', fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Create a twin y-axis for the Power (MW) actions
        ax1_twin = ax1.twinx()
        
        # Plot discharge as positive (red) and charge as negative (green) for easy reading
        ax1_twin.bar(range(T), discharge, label='Discharge (Sell)', color='red', alpha=0.6, width=0.8)
        ax1_twin.bar(range(T), -charge, label='Charge (Buy)', color='green', alpha=0.6, width=0.8)
        ax1_twin.set_ylabel('Power (MW)', color='dimgrey', fontweight='bold')
        ax1_twin.set_ylim([-max(charge)-5, max(discharge)+5]) # Add some breathing room

        # Combine legends for the top plot
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax1.set_title('Battery Arbitrage & Market Prices vs. Time', fontsize=14, fontweight='bold')

        # ==========================================
        # BOTTOM PLOT: State of Charge (SoC)
        # ==========================================
        # Use a step plot because SoC changes incrementally across the hour boundaries
        ax2.step(range(T + 1), soc, where='post', color='dodgerblue', linewidth=2.5, label='State of Charge')
        ax2.fill_between(range(T + 1), soc, step='post', color='dodgerblue', alpha=0.2)
        
        ax2.set_xlabel('Hour of the Day', fontweight='bold')
        ax2.set_ylabel('Energy (MWh)', fontweight='bold')
        ax2.set_xlim([0, T])
        ax2.set_xticks(range(0, T + 1, 2))
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.show()