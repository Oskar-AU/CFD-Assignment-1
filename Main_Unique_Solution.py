import numpy as np
import matplotlib.pyplot as plt

def solve_fvm(N, Pe, scheme='linear'):
    L = 1.0  # Domain length
    dx = L / N  # Grid spacing
    u = 1.0  # Velocity
    Gamma = u * L/Pe  # Diffusion coefficient

    b = np.zeros(N)
    A = np.zeros((N, N))

    Dw = Gamma / dx  # Diffusion term west
    De = Gamma / dx  # Diffusion term east
    
    # Choose advection scheme
    if scheme == 'linear':
        Fw, Fe = u / 2, u / 2  # Linear scheme
        # Apply boundary conditions (Dirichlet)
        # Right-hand side vector
        b[0] = (u+2*Dw)*0
        b[-1] = (-u+2*De)*1
        A[0, 0] = Fw + De + 2*Dw  # Left boundary (phi = 0 at x=0)
        A[0, 1] = Fe - De
        A[-1, -1] = -Fw + 2*De + Dw  # Right boundary (phi = 1 at x=L)
        A[-1, -2] = -Fe-Dw
    elif scheme == 'upwind':
        Fw, Fe = max(u, 0), min(u, 0)  # Upwind scheme
        # Apply boundary conditions (Dirichlet)
        # Right-hand side vector
        b[0] = (Fw+2*Dw)*0
        b[-1] = (-Fe+2*De)*1
        # Left hand side matrix
        A[0, 0] = Fw-Fe + De + 2*Dw  # Left boundary (phi = 0 at x=0)
        A[0, 1] = Fe - De
        A[-1, -1] = -Fe+Fw+2*De + Dw  # Right boundary (phi = 1 at x=L)
        A[-1, -2] = -Fw-Dw
    else:
        raise ValueError("Unknown scheme. Use 'linear' or 'upwind'.")

    # Coefficients     
    aW = -Fw-Dw
    aE = Fe-De
    aP = -aW-aE
    
    # Construct coefficient matrix A
    for i in range(1, N-1):
        A[i, i-1] = aW
        A[i, i] = aP
        A[i, i+1] = aE

    # Solve the system
    phi = np.linalg.solve(A, b)
    x = np.linspace(dx/2, L - dx/2, N)  # Compute cell center positions
    
    return x, phi

def exact_solution(x, Pe):
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

def plot_solutions():
    cases = [(100, 100), (20, 100), (20, -20)]  # (Grid size, Peclet number)
    
    plt.figure(figsize=(12, 5))
    for i, (N, Pe) in enumerate(cases, 1):
        x_linear, phi_linear = solve_fvm(N, Pe, 'linear')
        x_upwind, phi_upwind = solve_fvm(N, Pe, 'upwind')
        x_exact = np.linspace(0, 1, N)
        phi_exact = exact_solution(x_exact, Pe)
        
        plt.subplot(1, 3, i)
        plt.plot(x_exact, phi_exact, 'k-', label='Exact', linewidth=2)
        plt.plot(x_linear, phi_linear, 'b--', label='Linear', linewidth=1.5)
        plt.plot(x_upwind, phi_upwind, 'r-.', label='Upwind', linewidth=1.5)
        plt.xlabel('x')
        plt.ylabel('$\phi$')
        plt.title(f'N={N}, Pe={Pe}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Run the plot to visualize solutions
plot_solutions()

"""Truncation error plot"""
Pe = 20
L = 1
cases = np.logspace(1, 3, 50, dtype=int)
dx = L/cases
linear_errors = np.empty(len(cases))
upwind_errors = np.empty(len(cases))
fig, ax = plt.subplots()

for i, N in enumerate(cases):
    x_linear, phi_linear = solve_fvm(N, Pe, 'linear')
    x_upwind, phi_upwind = solve_fvm(N, Pe, 'upwind')
    phi_exact = exact_solution(x_linear, Pe)
    linear_errors[i] = np.sum(np.abs(phi_exact - phi_linear)) / N
    upwind_errors[i] = np.sum(np.abs(phi_exact - phi_upwind)) / N
ax.set_yscale('log')
ax.set_xscale('log')

linear_order_of_discretion_error = np.polyfit(np.log10(dx), np.log10(linear_errors), 1)
upwind_order_of_discretion_error = np.polyfit(np.log10(dx), np.log10(upwind_errors), 1)

ax.plot((dx[0], dx[-1]), 10**np.dot([[np.log10(dx[0]), 1], [np.log10(dx[-1]), 1]], linear_order_of_discretion_error), 
        label=f'Fit line ({linear_order_of_discretion_error[0]:.2f})', linewidth=2.0)
ax.plot((dx[0], dx[-1]), 10**np.dot([[np.log10(dx[0]), 1], [np.log10(dx[-1]), 1]], upwind_order_of_discretion_error), 
        label=f'Fit line ({upwind_order_of_discretion_error[0]:.2f})', linewidth=2.0)
ax.scatter(dx, linear_errors, label='Linear')
ax.scatter(dx, upwind_errors, label='Upwind')
ax.grid()
ax.legend()
ax.set_xlabel('Grid size [m]')
ax.set_ylabel('Truncation error [-]')
plt.show()