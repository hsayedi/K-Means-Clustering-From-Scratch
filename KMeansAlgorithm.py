#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 09:49:26 2019

@author: husnasayedi

Object-Oriented Programming Code 
Implemneting K-Means Clustering from Scratch

"""
import numpy as np
import pandas as pd
import random as rd
import pandas.api.types as ptypes
from matplotlib import pyplot as plt

def remove_outliers(df):
    """
    
    Parameters
    ----------
    df : pandas DataFrame
        original input data.

    Returns
    -------
    df : pandas DataFrame
        rermoves outliers ussing quantile method.

    """

    low = .10 #.05
    high = .90 #.95
    quant_df = df.quantile([low, high])
    for col_name in list(df.columns):
        if ptypes.is_numeric_dtype(df[col_name]):
             df = df[(df[col_name] > quant_df.loc[low, col_name]) & \
                     (df[col_name] < quant_df.loc[high, col_name])]
                 
    return df

def generate_data(filename):
    """
    
    Parameters
    ----------
    filename : path to CSV
        path and file name of data input.

    Returns
    -------
    home_staten : pandas DataFrame
        specificly filtered data sample of Staten Island with 2 features: 
            price_normalized and number_of_reviews.

    """

    df = pd.read_csv(filename)

    # Normalize the price by minimum_nights
    df['price_normalized'] = df['price']/df['minimum_nights']
    # Filter out listings with price of 0 
    df = df[df['price'] > 0]
    # Filter out listings that are available for 0 days out of the year
    df[df['availability_365'] < 10]

    # Entire Home / Apartment - Listings
    home = df[df.room_type == 'Entire home/apt']
    home_staten = home[home.neighbourhood_group == 'Staten Island']
    home_staten = home_staten[['price_normalized', 'number_of_reviews']]
    home_staten = remove_outliers(home_staten)   

    return home_staten



class KMeansAlgorithm(object):
    
    def __init__(self, df, K):
        self.data = df.values
        self.x_label = df.columns[0]
        self.y_label = df.columns[1]
        self.K = K                      # num clusters
        self.m = self.data.shape[0]     # num training examples
        self.n = self.data.shape[1]     # num of features
        self.result = {}
        self.centroids = np.array([]).reshape(self.n, 0)

    def init_random_centroids(self, data, K):
        """

        Parameters
        ----------
        data : numpy.ndarray
            DataFrame of 2 features converted into a numpy array.
        K : TYPE
            DESCRIPTION.

        Returns
        -------
        numpy.ndarray
            Centroids will be a (n x K) dimensional matrix.
            Each column will be one centroid for one cluster.

        """
        temp_centroids = np.array([]).reshape(self.n, 0)
        for i in range(K):
            rand = rd.randint(0, self.m-1)
            temp_centroids = np.c_[temp_centroids, self.data[rand]]
            
        return temp_centroids 


    def fit_model(self, num_iter):
        """
        
        Parameters
        ----------
        num_iter : int
            number of iterations until convergenc.

        Returns
        -------
        None.

        """
        
        # Initiate centroids randomly
        self.centroids = self.init_random_centroids(self.data, self.K)
        # Begin iterations to update centroids, compute and update Euclidean distances
        for i in range(num_iter):
            # First compute the Euclidean distances and store them in array
            EucDist = np.array([]).reshape(self.m, 0)
            for k in range(self.K):
                #print(k)
                dist = np.sum((self.data - self.centroids[:,k])**2, axis=1)
                #print(dist)
                EucDist = np.c_[EucDist, dist]
            # take the min distance
            min_dist = np.argmin(EucDist, axis=1) + 1
        
            # Begin iterations
            soln_temp = {} # temp dict which stores solution for one iteration - Y
        
            for k in range(self.K):
                soln_temp[k+1] = np.array([]).reshape(self.n, 0)
        
            for i in range(self.m):
                # regroup the data points based on the cluster index
                soln_temp[min_dist[i]] = np.c_[soln_temp[min_dist[i]], self.data[i]]
        
            for k in range(self.K):
                soln_temp[k+1] = soln_temp[k+1].T
               
            # Updating centroids as the new mean for each cluster
            for k in range(self.K):
                self.centroids[:,k] = np.mean(soln_temp[k+1], axis=0)
               
            self.result = soln_temp

    def plot_kmeans(self):
        """
        
        Returns
        -------
        plot
            final plot showing k clusters color coded with centroids.

        """
        # create arrays for colors and labels based on specified K
        colors = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)]) \
                  for i in range(self.K)]
        labels = ['cluster_' + str(i+1) for i in range(self.K)]

        fig1 = plt.figure(figsize=(5,5))
        ax1 = plt.subplot(111)
        # plot each cluster
        for k in range(self.K):
                ax1.scatter(self.result[k+1][:,0], self.result[k+1][:,1],
                                        c = colors[k], label = labels[k])
        # plot centroids
        ax1.scatter(self.centroids[0,:], self.centroids[1,:], #alpha=.5,
                                s = 300, c = 'lime', label = 'centroids')
        plt.xlabel(self.x_label) # first column of df
        plt.ylabel(self.y_label) # second column of df
        plt.title('Plot of K Means Clustering Algorithm')
        plt.legend()
        
        return plt.show(block=True)
        
        
    def predict(self):
        """
        
        Returns
        -------
        result
            minimum Euclidean distances from each centroid.
        centroids.T
            K centroids after n_iterations.
            
        """
        return self.result, self.centroids.T
    
    
    def plot_elbow(self):
        """
        
        Elbow Method:
        The elbow method will help us determine the optimal value for K. 
        Steps: 
        1) Use a range of K values to test which is optimal 
        2) For each K value, calculate Within-Cluster-Sum-of-Squares (WCSS) 
        3) Plot Num Clusters (K) x WCSS
        
        Returns
        -------
        plot
            elbow plot - k values vs wcss values to find optimal K value.

        """

        wcss_vals = np.array([])
        for k_val in range(1, self.K):
            results, centroids = self.predict()
            wcss=0
            for k in range(k_val):
                wcss += np.sum((results[k+1] - centroids[k,:])**2)
            wcss_vals = np.append(wcss_vals, wcss)
        # Plot K values vs WCSS values
        K_vals = np.arange(1, self.K)
        plt.plot(K_vals, wcss_vals)
        plt.xlabel('K Values')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        
        return plt.show(block=True)
        

def main():
    
    df_kmeans = generate_data('AB_NYC_2019.csv')
    kmeans = KMeansAlgorithm(df_kmeans, 4)
    kmeans.fit_model(10)
    kmeans.plot_kmeans()
    results, centroids = kmeans.predict()
    kmeans.plot_elbow()
    
if __name__ == '__main__':
  main()





