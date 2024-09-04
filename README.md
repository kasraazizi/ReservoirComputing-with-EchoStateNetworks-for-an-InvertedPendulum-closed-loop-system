# ReservoirComputing-with-EchoStateNetworks-for-an-InvertedPendulum-closed-loop-system
Echo State Networks in Reservoir Computing for a Custom Closed-Loop Inverted Pendulum

## **Project Goals**
1. Explore the application of Echo State Networks (ESNs) within the Reservoir Computing (RC) framework for controlling a custom closed-loop inverted pendulum system.
2. Demonstrate the effectiveness of ESN-based RC in modeling and controlling complex nonlinear dynamics.
3. Provide a practical example of implementing ESN-based RC using the PyTorch and ReservoirPy Python packages.

## **Project Description**
This project implements an ESN-based RC model for controlling a custom closed-loop inverted pendulum system. The notebook leverages PyTorch and ReservoirPy to explore how ESNs can be used to model and control nonlinear systems with high-dimensional dynamics. The project follows these steps:

1. **System Dynamics Definition:** Define the dynamics of the inverted pendulum, including its equations of motion and control inputs.
2. **Reservoir Setup:** Configure the ESN reservoir with appropriate hyperparameters, including the number of neurons, spectral radius, and input scaling factors.
3. **Training the ESN:** Train the ESN using a dataset generated from the inverted pendulum system, where the control signal serves as the input and the pendulum’s angle and angular velocity are the outputs.
4. **Performance Evaluation:** Assess the ESN’s performance by comparing its predictions to the actual system dynamics.

## **Implementation Files**
1.cart_pendulum_model.py: Defines the dynamics of the custom inverted pendulum system.

2. cart_pendulum_measurement.py: Defines the measurement for the custom closed-loop inverted pendulum system.

3. controller.py: Implements the closed-loop PID controller to control the inverted pendulum system.

## **Notebooks**
1. ESN-observation-cartpole.ipynb: Demonstrates the ESN-based RC model for observing of the latent states of the pendulum system.

2. ESN-prediction-cartpole.ipynb: Demonstrates the ESN-based RC model for predicting the future dynamics of the custom closed-loop inverted pendulum system.

## **Usage Instructions**
To run this notebook, follow these steps:

1. **Install Required Packages:** Ensure that the necessary Python packages are installed:
   ```bash
   pip install torch reservoirpy
   ```
2. **Download the Notebook:** Obtain the notebook from the provided link or repository.
3. **Run the Notebook:** Open the notebook in a Jupyter environment or any Python-compatible IDE.
4. **Execute the Code:** Run the cells in sequence to implement the ESN-based RC model for the custom closed-loop inverted pendulum system.
5. **Analyze Results:** Review the results to evaluate the ESN model's effectiveness in controlling the system.

## **References**
3. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
4. [ReservoirPy Documentation](https://reservoirpy.readthedocs.io/en/latest/)

## **Notes**
This notebook provides a foundational implementation of ESN-based RC for a custom closed-loop inverted pendulum system. Depending on your specific needs and problem domain, you may need to modify and extend the code.

---
