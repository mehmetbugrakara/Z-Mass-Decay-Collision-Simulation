import uproot3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, auc


class ParticleDataProcessor:
    def __init__(self, signal_path, background_path):
        """
        Initialize ParticleDataProcessor with paths to signal and background ROOT files.

        Args:
            signal_path (str): File path to the signal ROOT file.
            background_path (str): File path to the background ROOT file.
        """
        self.signal_path = signal_path
        self.background_path = background_path

    def load_signal_and_background(self):
        """
        Load signal and background data from ROOT files.

        Returns:
            tuple: A tuple containing the signal and background 'Delphes' trees.
                The first element is the signal 'Delphes' tree.
                The second element is the background 'Delphes' tree.
        """
        
        self.signal_path= "data/signal_delphes_T8_cv01.root"
        self.background_path = "data/bkg_sm_delphes.root"

        # Open the signal ROOT file and retrieve the 'Delphes' tree
        signal_file = uproot3.open(self.signal_path)
        signal_tree = signal_file['Delphes']

        # Open the background ROOT file and retrieve the 'Delphes' tree
        background_file = uproot3.open(self.background_path)
        background_tree = background_file['Delphes']

        return signal_tree, background_tree

    def process_data(self):
        """
        Process particle data from 'Delphes' trees for signal and background.

        Args:
            keys (uproot3.tree.TTree): 'Delphes' tree containing signal particle data.

        Returns:
            tuple: A tuple containing processed DataFrames for each particle type.
                Index 0: DataFrame for signal photon data
                Index 1: DataFrame for background photon data
                Index 2: DataFrame for signal muon data
                Index 3: DataFrame for background muon data
                Index 4: DataFrame for signal electron data
                Index 5: DataFrame for background electron data
                Index 6: DataFrame for signal MET data
                Index 7: DataFrame for background MET data
                Index 8: DataFrame for signal jet data
                Index 9: DataFrame for background jet data
        """
        keys, bg =self.load_signal_and_background()

        # Process photon data
        photon_arrs = keys.arrays(['Event.Number','Photon','Photon.PT','Photon.Eta','Photon.fBits','Photon.T','Photon.Phi'])
        photon_df = pd.DataFrame(photon_arrs)
        photon_df = photon_df.set_axis(['event_number','Photon','photon_pt','photon_eta','photon_fbits','photon_t','photon_phi'], axis="columns").reset_index(drop=True)
        photon_df = photon_df.explode(['photon_pt','photon_eta','photon_fbits','photon_t','photon_phi'],ignore_index=True).fillna(0)
        photon_df = photon_df.explode(['event_number'],ignore_index=True).fillna(0)
        
        # Process muon, electron, MET, jet data (similar steps as above)
        muon_arrs = keys.arrays(['Event.Number','Muon','Muon.PT','Muon.Eta','Muon.T','Muon.Phi'])
        muon_df = pd.DataFrame(muon_arrs)
        muon_df = muon_df.set_axis(['event_number','muon','muon_pt','muon_eta','muon_t','muon_phi'], axis="columns").reset_index(drop=True)
        muon_df = muon_df.explode(['muon_pt','muon_eta','muon_t','muon_phi'],ignore_index=True).fillna(0)
        muon_df = muon_df.explode(['event_number'],ignore_index=True).fillna(0)

        # Process electron data (similar steps as above)
        electron_arrs = keys.arrays(['Event.Number','Electron','Electron.PT','Electron.Eta','Electron.T','Electron.Phi'])
        electron_df = pd.DataFrame(electron_arrs)
        electron_df = electron_df.set_axis(['event_number','Electron','electron_pt','electron_eta','electron_t','electron_phi'], axis="columns").reset_index(drop=True)
        electron_df = electron_df.explode(['electron_pt','electron_eta','electron_t','electron_phi'],ignore_index=True).fillna(0)
        electron_df = electron_df.explode(['event_number'],ignore_index=True).fillna(0)

        # Process MET data (similar steps as above)
        met_arrays = keys.arrays(['Event.Number','MissingET','MissingET.fBits','MissingET.MET','MissingET.Eta','MissingET.Phi'])
        met_df = pd.DataFrame(met_arrays)
        met_df = met_df.set_axis(['event_number','MissingET','met_fBits','met_MET','met_Eta','met_phi'], axis="columns").reset_index(drop=True)
        met_df = met_df.explode(['met_fBits','met_MET','met_Eta','met_phi'],ignore_index=True)
        met_df = met_df.explode(['event_number'],ignore_index=True).fillna(0)

        # Process jet data (similar steps as above)
        jet_arrs = keys.arrays(['Event.Number','Jet','Jet.PT','Jet.Eta','Jet.T','Jet.Phi'])
        jet_df = pd.DataFrame(jet_arrs)
        jet_df = jet_df.set_axis(['event_number','Jet','jet_pt','jet_eta','jet_t','jet_phi'], axis="columns").reset_index(drop=True)
        jet_df = jet_df.explode(['jet_pt','jet_eta','jet_t','jet_phi'],ignore_index=True)
        jet_df = jet_df.explode(['event_number'],ignore_index=True).fillna(0)

        # Process background photon data (similar steps as above)
        bg_photon_arrs = bg.arrays(['Event.Number','Photon','Photon.PT','Photon.Eta','Photon.fBits','Photon.T','Photon.Phi'])
        bg_photon_df = pd.DataFrame(bg_photon_arrs)
        bg_photon_df = bg_photon_df.set_axis(['event_number','Photon','photon_pt','photon_eta','photon_fbits','photon_t','photon_phi'], axis="columns").reset_index(drop=True)
        bg_photon_df = bg_photon_df.explode(['photon_pt','photon_eta','photon_fbits','photon_t','photon_phi'],ignore_index=True).fillna(0)
        bg_photon_df = bg_photon_df.explode(['event_number'],ignore_index=True).fillna(0)

        # Process background muon, electron, MET, jet data (similar steps as above)
        bg_muon_arrs = bg.arrays(['Event.Number','Muon','Muon.PT','Muon.Eta','Muon.T','Muon.Phi'])
        bg_muon_df = pd.DataFrame(bg_muon_arrs)
        bg_muon_df = bg_muon_df.set_axis(['event_number','muon','muon_pt','muon_eta','muon_t','muon_phi'], axis="columns").reset_index(drop=True)
        bg_muon_df = bg_muon_df.explode(['muon_pt','muon_eta','muon_t','muon_phi'],ignore_index=True).fillna(0)
        bg_muon_df = bg_muon_df.explode(['event_number'],ignore_index=True).fillna(0)

        # Process background electron data (similar steps as above)
        bg_electron_arrs = bg.arrays(['Event.Number','Electron','Electron.PT','Electron.Eta','Electron.T','Electron.Phi'])
        bg_electron_df = pd.DataFrame(bg_electron_arrs)
        bg_electron_df = bg_electron_df.set_axis(['event_number','Electron','electron_pt','electron_eta','electron_t','electron_phi'], axis="columns").reset_index(drop=True)
        bg_electron_df = bg_electron_df.explode(['electron_pt','electron_eta','electron_t','electron_phi'],ignore_index=True).fillna(0)
        bg_electron_df = bg_electron_df.explode(['event_number'],ignore_index=True).fillna(0)

        # Process background MET data (similar steps as above)
        bg_met_arrs = bg.arrays(['Event.Number','MissingET','MissingET.fBits','MissingET.MET','MissingET.Eta','MissingET.Phi'])
        bg_met_df = pd.DataFrame(bg_met_arrs)
        bg_met_df = bg_met_df.set_axis(['event_number','MissingET','met_fBits','met_MET','met_Eta','met_phi'], axis="columns").reset_index(drop=True)
        bg_met_df = bg_met_df.explode(['met_fBits','met_MET','met_Eta','met_phi'],ignore_index=True).fillna(0)
        bg_met_df = bg_met_df.explode(['event_number'],ignore_index=True).fillna(0)

        # Process background jet data (similar steps as above)
        bg_jet_arrs = bg.arrays(['Event.Number','Jet','Jet.PT','Jet.Eta','Jet.T','Jet.Phi'])
        bg_jet_df = pd.DataFrame(bg_jet_arrs)
        bg_jet_df = bg_jet_df.set_axis(['event_number','Jet','jet_pt','jet_eta','jet_t','jet_phi'], axis="columns").reset_index(drop=True)
        bg_jet_df = bg_jet_df.explode(['jet_pt','jet_eta','jet_t','jet_phi'],ignore_index=True)
        bg_jet_df = bg_jet_df.explode(['event_number'],ignore_index=True).fillna(0)

        return photon_df, muon_df, electron_df, met_df, jet_df, bg_photon_df, bg_muon_df, bg_electron_df, bg_met_df, bg_jet_df
    
    def calculate_sum(self,df):
        """
        Calculate the sum of values in each event for a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing particle data.

        Returns:
            pandas.DataFrame: DataFrame with the sum of values for each event.
        """
        # Group the DataFrame by event_number and sum the values for each group
        df = df.groupby(['event_number']).sum().reset_index()
        return df
    
    def sum_calculate_sum(self):

        photon_df, muon_df, electron_df, met_df, jet_df, bg_photon_df, bg_muon_df, bg_electron_df, bg_met_df, bg_jet_df = self.process_data()
        sum_jet = self.calculation_sum(jet_df)
        sum_muon = self.calculation_sum(muon_df)
        sum_electron = self.calculation_sum(electron_df)
        sum_photon = self.calculation_sum(photon_df)
        sum_bg_jet = self.calculation_sum(bg_jet_df)
        sum_bg_muon = self.calculation_sum(bg_muon_df)
        sum_bg_electron = self.calculation_sum(bg_electron_df)
        sum_bg_photon = self.calculation_sum(bg_photon_df)

        return sum_jet, sum_muon, sum_electron, sum_photon, sum_bg_jet, sum_bg_muon, sum_bg_electron, sum_bg_photon
    
    def get_values_for_each_group(self,df, group_column,value_column, num_values=2):
    
        # Group the DataFrame by the specified column
        grouped_df = df.groupby(group_column)
        # Initialize an empty dictionary to store results
        result_dict = {}

        # Iterate over each group
        for group_name, group_df in grouped_df:
            group_values = {}

            # Iterate over the specified number of values
            for i in range(num_values):
                # Grup boyutunu kontrol et
                if i < len(group_df):
                    column_name = f'{value_column}_{i}'
                    group_values[column_name] = group_df.iloc[i][value_column]
                else:
                    # Grup boyutu num_values'dan küçükse, eksik değerleri None olarak ayarla
                    column_name = f'{value_column}_{i}'
                    group_values[column_name] = None

            result_dict[group_name] = group_values

            # Convert the result dictionary to a DataFrame and reset the index
            result_df = pd.DataFrame.from_dict(result_dict, orient='index').reset_index()


        return result_df
    
    def eta_pt_phi_values_for_each_group(self):

        photon_df, muon_df, electron_df, met_df, jet_df, bg_photon_df, bg_muon_df, bg_electron_df, bg_met_df, bg_jet_df = self.process_data()
        muon_etas_df = self.get_values_for_each_group(muon_df, 'event_number', 'muon_eta')
        muon_pts_df = self.get_values_for_each_group(muon_df, 'event_number', 'muon_pt')
        muon_phis_df = self.get_values_for_each_group(muon_df, 'event_number', 'muon_phi')
        electron_etas_df= self.get_values_for_each_group(electron_df, 'event_number', 'electron_eta')
        electron_pts_df= self.get_values_for_each_group(electron_df, 'event_number', 'electron_pt')
        electron_phis_df= self.get_values_for_each_group(electron_df, 'event_number', 'electron_phi')
        bg_muon_etas_df = self.get_values_for_each_group(bg_muon_df, 'event_number', 'muon_eta')
        bg_muon_pts_df = self.get_values_for_each_group(bg_muon_df, 'event_number', 'muon_pt')
        bg_muon_phis_df = self.get_values_for_each_group(bg_muon_df, 'event_number', 'muon_phi')
        bg_electron_etas_df= self.get_values_for_each_group(bg_electron_df, 'event_number', 'electron_eta')
        bg_electron_pts_df= self.get_values_for_each_group(bg_electron_df, 'event_number', 'electron_pt')
        bg_electron_phis_df = self.get_values_for_each_group(bg_electron_df, 'event_number', 'electron_phi')

        muon_fs=muon_etas_df.merge(muon_pts_df,on=['index'],how='left').merge(muon_phis_df,on=['index'],how='left')
        electron_fs=electron_etas_df.merge(electron_pts_df,on=['index'],how='left').merge(electron_phis_df,on=['index'],how='left')
        bg_muon_fs=bg_muon_etas_df.merge(bg_muon_pts_df,on=['index'],how='left').merge(bg_muon_phis_df,on=['index'],how='left')
        bg_electron_fs=bg_electron_etas_df.merge(bg_electron_pts_df,on=['index'],how='left').merge(bg_electron_phis_df,on=['index'],how='left')


        muon_fs.rename(columns={'index':'event_number'},inplace=True)
        electron_fs.rename(columns={'index':'event_number'},inplace=True)
        bg_muon_fs.rename(columns={'index':'event_number'},inplace=True)
        bg_electron_fs.rename(columns={'index':'event_number'},inplace=True)

        bg_fs=bg_muon_fs.merge(bg_electron_fs,on=['event_number'],how='left')
        sg_fs=muon_fs.merge(electron_fs,on=['event_number'],how='left')

        return bg_fs, sg_fs
    
    def merge_data_frames(self):
        """
        Merge different data frames containing background and signal data.
        
        Args:
        - sum_bg_jet (DataFrame): DataFrame containing background jet data
        - sum_bg_muon (DataFrame): DataFrame containing background muon data
        - sum_bg_electron (DataFrame): DataFrame containing background electron data
        - sum_bg_photon (DataFrame): DataFrame containing background photon data
        - bg_fs (DataFrame): DataFrame containing background data for final state
        - sum_jet (DataFrame): DataFrame containing signal jet data
        - sum_muon (DataFrame): DataFrame containing signal muon data
        - sum_electron (DataFrame): DataFrame containing signal electron data
        - sum_photon (DataFrame): DataFrame containing signal photon data
        - sg_fs (DataFrame): DataFrame containing signal data for final state
        
        Returns:
        - final_df (DataFrame): Concatenated DataFrame containing merged background and signal data
        - target (Series): Series containing the target variable indicating background (0) or signal (1)
        """

        sum_jet, sum_muon, sum_electron, sum_photon, sum_bg_jet, sum_bg_muon, sum_bg_electron, sum_bg_photon = self.sum_calculate_sum()
        bg_fs, sg_fs = self.eta_pt_phi_values_for_each_group()
        # Merging background data frames
        total_sum_bg_df = sum_bg_jet.merge(sum_bg_muon, on=['event_number'], how='left') \
                                    .merge(sum_bg_electron, on=['event_number'], how='left') \
                                    .merge(sum_bg_photon, on=['event_number'], how='left') \
                                    .merge(bg_fs, on=['event_number'], how='left')
        
        # Merging signal data frames
        total_sum_df = sum_jet.merge(sum_muon, on=['event_number'], how='left') \
                            .merge(sum_electron, on=['event_number'], how='left') \
                            .merge(sum_photon, on=['event_number'], how='left') \
                            .merge(sg_fs, on=['event_number'], how='left')
        
        # Removing duplicate columns
        total_sum_df = total_sum_df.loc[:, ~total_sum_df.columns.duplicated()].copy()
        total_sum_bg_df = total_sum_bg_df.loc[:, ~total_sum_bg_df.columns.duplicated()].copy()
        
        # Adding class labels
        total_sum_df['Class'] = 1
        total_sum_bg_df['Class'] = 0
        
        # Concatenating data frames
        final_df = pd.concat([total_sum_df, total_sum_bg_df], axis=0)
        
        # Extracting target variable
        target = final_df['Class']
    
        return final_df, target
    

    def remove_outliers_multi_columns(self, df, columns, threshold=1.5):
        """
        DataFrame içinde belirli sütunlardaki aykırı değerleri kaldırır.

        Parametreler:
        - df: DataFrame
        - columns: Aykırı değerleri kaldırmak istediğiniz sütun adlarını içeren liste (varsayılan: None, tüm sütunlarda arama yapar)
        - threshold: Aykırı değer tespiti için kullanılacak eşik değeri (varsayılan: 1.5)

        Dönüş:
        - cleaned_df: Aykırı değerleri kaldırılmış DataFrame
        """

        cleaned_df = df.copy()

        for column in columns:
            z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
            outliers = cleaned_df[z_scores > threshold]
            cleaned_df = cleaned_df[z_scores <= threshold]

        return cleaned_df,outliers
    
    def calculate_z_mass(self,row):
        lepton1_p = row['muon_pt_0'] / np.sinh(row['muon_eta_0'])
        lepton2_p = row['muon_pt_1'] / np.sinh(row['muon_eta_1'])

        total_p = np.array([lepton1_p * np.cos(row['muon_phi_0']) + lepton2_p * np.cos(row['muon_phi_1']),
                            lepton1_p * np.sin(row['muon_phi_0']) + lepton2_p * np.sin(row['muon_phi_1']),
                            lepton1_p * np.sinh(row['muon_eta_0']) + lepton2_p * np.sinh(row['muon_eta_1']),
                            lepton1_p * np.cosh(row['muon_eta_0']) + lepton2_p * np.cosh(row['muon_eta_1'])])

        z_mass = np.sqrt(total_p[3]**2 - np.sum(total_p[:3]**2))
        return z_mass
    
    def visualize_features(self, final_df, target, num_columns=3, figsize=(15, 22), dpi=100, save_path=None):
        """
        Visualize histograms of features for background (B) and signal (S) classes.

        Args:
        - final_df (DataFrame): DataFrame containing features and target variable
        - target (Series): Series containing the target variable indicating background (0) or signal (1)
        - num_columns (int): Number of columns for the subplot grid (default is 3)
        - figsize (tuple): Figure size (width, height) in inches (default is (15, 22))
        - dpi (int): Dots per inch for figure resolution (default is 100)
        - save_path (str): File path to save the visualization (default is None)

        Returns:
        - None
        """
        # Get features names and number of features
        features_name = final_df.columns
        num_features = final_df.shape[1]

        # Calculate number of rows needed for subplot grid
        num_rows = int(math.ceil(num_features / num_columns))

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize, dpi=dpi)

        # Plot histograms for each feature
        for feature_index, ax in enumerate(axes.flat):
            if feature_index < num_features:
                final_df[target == 0].iloc[:, feature_index].hist(ax=ax, color='b', alpha=0.2, density=False, label='B')
                final_df[target == 1].iloc[:, feature_index].hist(ax=ax, color='r', alpha=0.2, density=False, label='S')
                ax.set_title(features_name[feature_index])
                ax.legend()
                ax.set_yscale('log')

        # Remove empty subplots
        for feature_index in range(num_features, num_rows * num_columns):
            fig.delaxes(axes.flatten()[feature_index])

        # Adjust layout and save the plot if save_path is provided
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)

    def visualize_feature_importance(self, X, y, n_estimators=80, random_state=0, figsize=(15, 22), dpi=100, save_path=None):
        """
        Visualize feature importance using RandomForestRegressor.

        Args:
        - X (DataFrame): DataFrame containing features
        - y (Series): Series containing the target variable
        - n_estimators (int): Number of trees in the forest (default is 80)
        - random_state (int): Random seed for reproducibility (default is 0)
        - figsize (tuple): Figure size (width, height) in inches (default is (15, 22))
        - dpi (int): Dots per inch for figure resolution (default is 100)

        Returns:
        - None
        """
        # Fit RandomForestRegressor
        rf_regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf_regressor.fit(X, y)

        # Get feature importance
        feature_importance = rf_regressor.feature_importances_
        feature_name = X.columns

        # Create DataFrame for feature importance
        feature_importance_df = pd.DataFrame({'Feature': feature_name, 'Importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plot histogram of feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xticks(rotation='vertical')
        plt.title('Feature importance')
        if save_path:
            plt.savefig(save_path)

    def visualize_correlation_matrix(self, data, figsize=(50, 50), cmap='coolwarm', annot=True, fmt=".2f", save_path=None):
        """
        Visualize correlation matrix using a heatmap.

        Args:
        - data (DataFrame): DataFrame containing the data
        - figsize (tuple): Figure size (width, height) in inches (default is (50, 50))
        - cmap (str): Colormap for the heatmap (default is 'coolwarm')
        - annot (bool): Whether to annotate the heatmap with correlation values (default is True)
        - fmt (str): String formatting code to use when adding annotations (default is ".2f")

        Returns:
        - None
        """
        # Calculate correlation matrix
        correlation_matrix = data.corr()

        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=fmt)
        plt.title('Korelasyon Matrisi')
        if save_path:
            plt.savefig(save_path)

    def train_test_models_and_evaluate(self, cleaned_df):
        """
        Train and evaluate multiple classification models.

        Args:
        - cleaned_df (DataFrame): DataFrame containing the cleaned data

        Returns:
        - model_accuracies (dict): Dictionary containing model accuracies
        """
        # Veriyi eğitim ve test setlerine bölelim
        cleaned_df.sample(frac=1)
        X = cleaned_df.drop(columns=['Class'])
        y = cleaned_df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train=X_train.astype(float)
        X_test=X_test.astype(float)

        # Logistic Regression
        logreg_model = LogisticRegression()
        logreg_model.fit(X_train, y_train)
        y_pred_logreg = logreg_model.predict(X_test)
        accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

        # Decision Tree
        tree_model = DecisionTreeClassifier()
        tree_model.fit(X_train, y_train)
        y_pred_tree = tree_model.predict(X_test)
        accuracy_tree = accuracy_score(y_test, y_pred_tree)

        # Random Forest
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)

        # XGBoost
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

        # CatBoost
        catboost_model = CatBoostClassifier(iterations=100, depth=10, learning_rate=0.05, loss_function='Logloss')
        catboost_model.fit(X_train, y_train)
        y_pred_catboost = catboost_model.predict(X_test)
        accuracy_catboost = accuracy_score(y_test, y_pred_catboost)

        # Model accuracies dictionary
        model_accuracies = {
            'Logistic Regression': accuracy_logreg,
            'Decision Tree': accuracy_tree,
            'Random Forest': accuracy_rf,
            'XGBoost': accuracy_xgb,
            'CatBoost': accuracy_catboost
        }

        return model_accuracies, X, y
    def plot_roc_curve(self, model, X_test, y_test, model_name, save_path=None):
        """
        Plot the ROC curve for a given model.

        Args:
        - model: The trained classification model
        - X_test (DataFrame): Testing features
        - y_test (Series): Testing target
        - model_name (str): Name of the model for labeling the plot

        Returns:
        - None
        """
        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve ({})'.format(model_name))
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path)

    def calculate_significance(self, cleaned_df):
        """
        Calculate the significance using the signal and background events in the cleaned DataFrame.

        Args:
        - cleaned_df (DataFrame): DataFrame containing the cleaned data

        Returns:
        - significance (float): Significance value
        """
        # Count the number of background events with muon_pt_0 greater than 0
        background_signal = cleaned_df[(cleaned_df['Class'] == 0) & (cleaned_df['muon_pt_0'] > 0)].shape[0]
        
        # Count the number of signal events
        signal = cleaned_df[cleaned_df['Class'] == 1].shape[0]
        
        # Calculate significance
        significance = signal / ((signal + background_signal) ** 0.5)
        
        return significance
    
    def main(self):

        final_df, target = self.merge_data_frames()
        cleaned_df, outlier = self.remove_outliers_multi_columns(final_df, columns=['jet_pt', 'jet_eta', 'jet_t', 'jet_phi',
       'photon_pt', 'photon_eta', 'photon_fbits', 'photon_t', 'photon_phi',
       'muon_pt', 'muon_eta', 'muon_t', 'muon_phi', 'electron_pt',
       'electron_eta', 'electron_t', 'electron_phi'])
        self.visualize_features(final_df,target,save_path="save_path.png")
        model_accuracies , X, y =self.train_test_models_and_evaluate(cleaned_df)
        self.visualize_feature_importance(X, y, save_path="save_path.png")
        self.visualize_correlation_matrix(cleaned_df, save_path="save_path.png")
        significance=self.calculate_significance(cleaned_df)

    if __name__ == "__main__":
        main()


        



        



