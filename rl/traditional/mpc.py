import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.optimize as optimize
from matplotlib.patches import Rectangle, Circle
import time

class CartPoleSystem:
    def __init__(self, m_cart=1.0, m_pole=0.1, l=1.0, g=9.8):
        """
        Initialize cart-pole system parameters
        
        Args:
            m_cart: mass of the cart (kg)
            m_pole: mass of the pole (kg)
            l: length of the pole (m)
            g: gravitational acceleration (m/s^2)
        """
        self.m_cart = m_cart  # Mass of the cart (kg)
        self.m_pole = m_pole  # Mass of the pole (kg)
        self.l = l            # Length of the pole (m)
        self.g = g            # Gravity (m/s^2)
        
        # State: [x, x_dot, theta, theta_dot]
        # x: position of cart
        # x_dot: velocity of cart
        # theta: angle of pole (0 is upright)
        # theta_dot: angular velocity of pole
        self.state = np.array([-.75, 0.5, 0.8, -1.0])  # Cart left, moving, pole falling

        
        # Time step for simulation
        self.dt = 0.02  # 20ms

    def dynamics(self, state, u):
        """
        Compute the dynamics of the cart-pole system
        
        Args:
            state: current state [x, x_dot, theta, theta_dot]
            u: control input (force applied to cart)
            
        Returns:
            next_state: next state after applying dynamics
        """
        x, x_dot, theta, theta_dot = state
        
        # Calculate the derivatives
        # Reference: https://underactuated.mit.edu/acrobot.html
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        m1 = self.m_cart
        m2 = self.m_pole
        l = self.l
        g = self.g
        
        # Compute acceleration components
        den = m1 + m2 * sin_theta**2
        
        # Acceleration of cart
        x_ddot = (u + m2 * sin_theta * (l * theta_dot**2 + g * cos_theta)) / den
        
        # Angular acceleration of pole
        theta_ddot = (u * cos_theta + m2 * l * theta_dot**2 * sin_theta * cos_theta + 
                      (m1 + m2) * g * sin_theta) / (-l * den)
        
        # Compute next state using Euler integration
        next_state = np.zeros_like(state)
        next_state[0] = x + x_dot * self.dt
        next_state[1] = x_dot + x_ddot * self.dt
        next_state[2] = theta + theta_dot * self.dt
        next_state[3] = theta_dot + theta_ddot * self.dt
        
        return next_state
    
    def step(self, u):
        """
        Step the simulation forward by applying control u
        
        Args:
            u: control input (force applied to cart)
            
        Returns:
            next_state: the state after applying control
        """
        self.state = self.dynamics(self.state, u)
        return self.state
    
    def reset(self, state=None):
        """Reset the system to a specified or default state"""
        if state is None:
            self.state = np.array([0.0, 0.0, 0.1, 0.0])
        else:
            self.state = np.array(state)
        return self.state


class MPCController:
    def __init__(self, system, Q=None, R=None, horizon=10, constraints=None):
        """
        Initialize MPC controller for cart-pole system
        
        Args:
            system: CartPoleSystem object
            Q: State cost matrix (4x4)
            R: Control cost matrix (1x1)
            horizon: Prediction horizon
            constraints: Dictionary of constraints
        """
        self.system = system
        
        # Default cost matrices if not provided
        if Q is None:
            # Penalize position, angle, and their derivatives
            self.Q = np.diag([1.0, 0.1, 10.0, 1.0])
        else:
            self.Q = Q
            
        if R is None:
            # Penalize control effort
            self.R = np.array([[0.1]])
        else:
            self.R = R
            
        self.horizon = horizon
        
        # Default constraints
        if constraints is None:
            self.constraints = {
                'u_min': -10.0,     # Minimum force
                'u_max': 10.0,      # Maximum force
                'x_min': -2.0,      # Minimum cart position
                'x_max': 2.0,       # Maximum cart position
            }
        else:
            self.constraints = constraints
    
    def objective(self, u_sequence, current_state, reference):
        """
        MPC objective function to minimize
        
        Args:
            u_sequence: Sequence of control inputs over horizon
            current_state: Current state of the system
            reference: Reference state to track
            
        Returns:
            cost: Total cost over horizon
        """
        cost = 0.0
        state = current_state.copy()
        
        # Loop over the prediction horizon
        for i in range(self.horizon):
            # Get control input for this step
            u = u_sequence[i]
            
            # State tracking cost
            state_error = state - reference
            cost += state_error.T @ self.Q @ state_error
            
            # Control cost
            cost += u * self.R * u
            
            # Simulate next state
            state = self.system.dynamics(state, u)
            
        return cost
    
    def optimize(self, current_state, reference):
        """
        Solve the MPC optimization problem
        
        Args:
            current_state: Current system state
            reference: Reference state to track
            
        Returns:
            u_optimal: Optimal control input for current step
        """
        # Initial guess: zero control for all steps in horizon
        u_init = np.zeros(self.horizon)
        
        # Bounds for control inputs
        bounds = [(self.constraints['u_min'], self.constraints['u_max'])] * self.horizon
        
        # Optimize control sequence
        result = optimize.minimize(
            lambda u: self.objective(u, current_state, reference),
            u_init,
            method='SLSQP',
            bounds=bounds
        )
        
        # Return only the first control input from the optimal sequence
        return result.x[0]


class CartPoleVisualizer:
    def __init__(self, cart_pole, controller=None, reference=None):
        """
        Initialize visualizer for cart-pole system
        
        Args:
            cart_pole: CartPoleSystem object
            controller: Controller object (optional)
            reference: Reference state for control (optional)
        """
        self.cart_pole = cart_pole
        self.controller = controller
        
        if reference is None:
            # Default reference: balance at center
            self.reference = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.reference = reference
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Cart-Pole System with MPC Control')
        self.ax.set_xlabel('Position (m)')
        
        # Create cart and pole objects
        self.cart_width = 0.5
        self.cart_height = 0.25
        self.cart = Rectangle(
            (0 - self.cart_width/2, 0 - self.cart_height/2),
            self.cart_width,
            self.cart_height,
            fill=True,
            color='blue',
            ec='black'
        )
        self.ax.add_patch(self.cart)
        
        # Pole (line)
        pole_x = [0, self.cart_pole.l * np.sin(self.cart_pole.state[2])]
        pole_y = [0, self.cart_pole.l * np.cos(self.cart_pole.state[2])]
        self.pole, = self.ax.plot(pole_x, pole_y, 'k-', lw=3)
        
        # Pole tip (circle)
        self.pole_tip = Circle(
            (pole_x[1], pole_y[1]),
            0.1,
            fill=True,
            color='red'
        )
        self.ax.add_patch(self.pole_tip)
        
        # Text for state information
        self.state_text = self.ax.text(-4.8, 1.7, '', fontsize=10)
        self.control_text = self.ax.text(-4.8, 1.5, '', fontsize=10)
        
        # Store state history for plotting
        self.time_history = []
        self.state_history = []
        self.control_history = []
        self.start_time = time.time()
    
    def update(self, frame):
        """Update function for animation"""
        # Apply control if controller is provided
        if self.controller:
            u = self.controller.optimize(self.cart_pole.state, self.reference)
            self.control_history.append(u)
        else:
            u = 0
            self.control_history.append(u)
        
        if not hasattr(self, "logged_data"):
            self.logged_data = []

        # Log current state and action
        self.logged_data.append((self.cart_pole.state.copy(), u))

        
        # Step the system
        next_state = self.cart_pole.step(u)
        
        # Store state for plotting
        self.time_history.append(time.time() - self.start_time)
        self.state_history.append(next_state.copy())
        
        # Update cart position
        cart_x = next_state[0] - self.cart_width/2
        self.cart.set_xy((cart_x, -self.cart_height/2))
        
        # Update pole
        pole_x = [next_state[0], next_state[0] + self.cart_pole.l * np.sin(next_state[2])]
        pole_y = [0, self.cart_pole.l * np.cos(next_state[2])]
        self.pole.set_data(pole_x, pole_y)
        
        # Update pole tip
        self.pole_tip.center = (pole_x[1], pole_y[1])
        
        # Update text
        state_str = f'x: {next_state[0]:.2f} m, θ: {(next_state[2] * 180/np.pi):.1f}°\n'
        state_str += f'v: {next_state[1]:.2f} m/s, ω: {(next_state[3] * 180/np.pi):.1f}°/s'
        self.state_text.set_text(state_str)
        
        self.control_text.set_text(f'Control: {u:.2f} N')
        
        return self.cart, self.pole, self.pole_tip, self.state_text, self.control_text
    
    def animate(self, frames=200, interval=20):
        """Run animation"""
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            interval=interval,
            blit=True
        )
        plt.show()


def main():
    # Create cart-pole system
    cart_pole = CartPoleSystem()
    cart_pole.reset([0, 0.0, 0, 0.0])  # Zero state: center + upright

    
    # Define cost matrices for MPC
    Q = np.diag([1.0, 0.1, 10.0, 0.1])  # State cost (position, velocity, angle, angular velocity)
    R = np.array([[0.01]])              # Control cost
    
    # Create MPC controller
    mpc = MPCController(
        system=cart_pole,
        Q=Q,
        R=R,
        horizon=20,
        constraints={
            'u_min': -10.0,
            'u_max': 10.0,
            'x_min': -1.0,
            'x_max': 1.0,
        }
    )
    
    # Define reference state (upright position at center)
    reference = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Initialize visualizer
    visualizer = CartPoleVisualizer(cart_pole, mpc, reference)
    
    # Run animation
    print("Running cart-pole MPC simulation...")
    print("Initial state: Slightly tilted pole")
    print("Goal: Balance the pole in upright position")
    visualizer.animate(frames=500)
    # Save logged data (state, action)
    import pickle

    log_data = getattr(visualizer, "logged_data", [])
    with open("mpc_log.pkl", "wb") as f:
        pickle.dump(log_data, f)

    print(f"Saved {len(log_data)} state-action pairs to mpc_log.pkl")



if __name__ == "__main__":
    main()