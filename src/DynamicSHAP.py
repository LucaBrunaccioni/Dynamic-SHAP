import numpy as np
import shap 
import matplotlib.pyplot as plt
import os

class DSHAP:
    """
    Compute and visualize D-SHAP importance profiles for multivariate
    batch-process data.

    The class evaluates SHAP values for a trained classification model,
    aggregates their absolute magnitudes across batches, and produces
    time-resolved (continuous) and cumulative D-SHAP visualizations.

    Parameters
    ----------
    model : object
        Trained machine-learning model compatible with ``shap.Explainer``.

    X : ndarray of shape (n_batches, n_sensors * n_timesteps)
        Multiway-PLS-like feature matrix containing sensor trajectories for all
        batches.

    dimensions : tuple[int, int, int]
        Original data dimensions:
        ``(n_batches, n_sensors, n_timesteps)``.

    sensor_names : list[str]
        Names of the sensors corresponding to the feature trajectories.

    onspec_class : int
        Class label (0 or 1) corresponding to the on-spec category.

    dir_name : str, optional
        Directory used to store generated figures.
        Default is ``"DSHAP_fig"``.

    Notes
    -----


    Methods
    -------
    fit()
        1. Create the output directory if it does not exist.
        2. Build a SHAP explainer.
        3. Compute SHAP values.
        4. Construct the sensor-by-time D-SHAP matrix used for plotting.

    plot_continuous_DSHAP()
        Generate and save a D-SHAP profile for each sensor over time.

    plot_cumulative_DSHAP(bar_width=1.0)
        Generate and save a cumulative D-SHAP importance plot together
        with time-step contribution bars.
    """
    
    def __init__(self, model, X, dimensions, sensor_names, onspec_class, dir_name="DSHAP_fig"):

        # Instantiate class
        self.model = model
        self.X = X
        self.dimensions = dimensions
        self.sensor_names = sensor_names
        self.onspec_class = onspec_class
        self.dir_name = dir_name

        # Checking for errors in input dimensions
        I = dimensions[0]
        J = dimensions[1]
        K = dimensions[2]

        if X.shape != (I, J*K):
            raise ValueError(
                f"Expected X shape {(I, J*K)}, got {X.shape}"
            )

        if len(sensor_names) != J:
            raise ValueError(
                "Sensor_names length must equal number of sensors"
            )

    def fit(self):
        
        # Build folder for figures
        self._build_figures_folder()

        # Activate preprocessing steps
        self._preprocessing()

    def _preprocessing(self):
        
        # Preprocess input parameters 
        self._build_explainer()
        self._calculate_shap_values()
        self._build_shap_array_for_plotting()
    
    def _build_figures_folder(self):

        # Get current working directory
        cwd = os.getcwd()

        # Set target directory path
        target_dir = os.path.join(cwd, self.dir_name)

        # Check presence of figures folder
        if not os.path.exists(target_dir):

            # Create figures folder if it does not exists
            os.mkdir(target_dir)

        self.target_dir = target_dir

    def _build_explainer(self):

        # Build explainer
        explainer = shap.Explainer(model=self.model)
        self.explainer = explainer

    def _calculate_shap_values(self):

        # Calculate SHAP values
        shaps = self.explainer.shap_values(X=self.X)[:, :, self.onspec_class]

        # Average absolute SHAP values across all batches (axis 0)
        shaps = np.mean(np.abs(shaps), axis=0)
        self.shaps = shaps

    def _build_shap_array_for_plotting(self):
        
        # Extract dimensions 
        J = self.dimensions[1] # Number of sensors
        K = self.dimensions[2] # Number of timesteps

        # Reshape array for plotting purposes
        self.shaps = self.shaps.reshape(K, J).T


    def plot_continuous_DSHAP(self):
        
        # Plot Continuous DSHAP for each sensor
        J = self.dimensions[1]
        K = self.dimensions[2]
        
        # Set x-axis values
        k_plot = np.arange(K)

        # Set y-axis ranges
        y_max = np.max(self.shaps)
        y_min = np.min(self.shaps)

        # Plot each DSHAP profile individually
        for j in range(J):

            plt.figure()

            plt.plot(k_plot, self.shaps[j, :]) 
            plt.title(self.sensor_names[j], fontsize=18, fontweight='bold')
            plt.ylabel("DSHAP Value", fontsize=16)
            plt.xlabel("Time Level", fontsize=16)
            plt.ylim(y_min, y_max)
            plt.grid(axis='x')
            plt.tight_layout()

            fig_name = os.path.join(
                self.target_dir,
                f"{self.sensor_names[j]}.png"
            )
            plt.savefig(fig_name)
            plt.close()

    def plot_cumulative_DSHAP(self, bar_width=1.):

        # Memory allocation
        K = self.dimensions[2]
        cumulative_shap = np.zeros(K)

        # Calculate cumulative DSHAP
        tot_sum_shap_matrix = np.sum(self.shaps)
        for k in range(K):
            cumulative_shap[k] = np.sum(self.shaps[:,:k+1]) / tot_sum_shap_matrix

        # Calculate DSHAP bar plot
        shap_bar = np.zeros_like(cumulative_shap)
        for k in range(K-1,0,-1):
            shap_bar[k] = cumulative_shap[k] - cumulative_shap[k-1]

        # Plotting
        fig, ax1 = plt.subplots()
        k_plot = np.arange(K)

        ax1.plot(k_plot, cumulative_shap, color='blue', linewidth=2.5)
        ax1.set_xlabel("Time Level",  fontsize=16)
        ax1.set_ylabel("Cumulative D-SHAP Score", color='blue',  fontsize=16)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(axis='x', zorder=0)

        ax2 = ax1.twinx()
        ax2.bar(k_plot, shap_bar, width=bar_width, color='red', alpha=0.5)
        ax2.set_ylabel("D-SHAP Bar Score", color='red',  fontsize=16)
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title("Cumulative D-SHAP Score", fontweight='bold', fontsize=18)
        plt.tight_layout()
        fig_name = os.path.join(
            self.target_dir,
            "Cumulative_DSHAP.png"
        )
        plt.savefig(fig_name)
        plt.close()