import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from matplotlib import colors
from matplotlib.widgets import Slider, Button, TextBox
import itertools
import pandas as pd
from scipy.optimize import curve_fit

class NpfEventAnalyzer:
    
    def __init__(self):
        self.fontsizes = 8
        return

    def combine_sizedist(self, times, diams, datas, time_resolution=1/1440.):
        """ A utility function to combine number-size distributions 

        result = EventAnalyzer.combine_sizedist([time_vector_dist_1, time_vector_dist_2, ..., time_vector_dist_n],
                                                [diam_vector_dist_1, diam_vector_dist_2, ..., diam_vector_dist_n],
                                                [data_matrix_dist_1, data_matrix_dist_2, ..., data_matrix_dist_n],
                                                time_resolution = 5/1440.)

        where for the dist_i
        time_vector_dist_i: 1-D array with length n and unit of days
        diam_vector_dist_i: 1-D arrays with length m and unit of nm
        data_matrix_dist_i: 2-D array with n rows and m columns containing the dNdlogDp in units of cm-3
        time_resolution: is the desired time resolution in days for the result, e.g. 1/1440. = 1 min (default)

        result is a list where
        result[0]: unified time vector 1-d array
        result[1]: unified diameter vector 1-d array
        result[2]: unified data matrix 2-d array

        """
        
        dfs = []
        for i in range(0, len(times)):
            df = pd.DataFrame(columns = diams[i], index = times[i] - np.floor(times[i]), data = datas[i])
            df = df.reindex(np.arange(0,1,time_resolution),method='nearest')
            dfs.append(df)
            
        sizedist = pd.concat(dfs,axis=1).sort_index(axis=1)
        
        return [sizedist.index.values, sizedist.columns.values, sizedist.values]

    def analyze_par(self, par, temp=273.15, pres=101325.0):
        """ Function that initializes the npf event analysis for particles

        EventAnalyzer.analyze_par(par,temp=273.15, pres=101325.0)
        
        par[0]: time vector, 1-D array, length n, unit days
        par[1]: diameter vector, 1-D array, length m, unit nm
        par[2]: particle number-size distribution (dNdlogDp), 2-D array, n-by-m, unit cm-3 

        temp[:,0] = time vector associated with the temperatures, unit days
        temp[:,1] = temperatures in Kelvin
        Also can be single float value, default 273.15 K

        pres[:,0] = time vector associated with the pressures, unit days
        pres[:,1] = pressure in Pascals
        Also can be single float value, default 101325.0 Pa

        """
        
        # Close any existing figures
        plt.close('all')
        
        # Turn "particle mode" on
        self.ion_mode = 0
        self.particle_mode = 1
        
        self.par = par
        self.temp = temp
        self.pres = pres
        
        self.__init_par_fig()
        self.__init_meteo()
        self.__init_variables()
        self.__init_polygons()
        self.__init_plots()
        self.__init_sliders()
        self.__init_buttons()
        self.__init_textboxes()
        
        plt.show()

    def analyze_ion(self, par, ion1, ion2, temp=273.15, pres=101325.0):
        """ Function that initializes the event analysis for ions

        This function can be used to calculate GR and J for 
        the polarity of ions given in ion1, the opposite polarity ions are 
        given in ion2. 

        EventAnalyzer.analyze_ion(par, ion1, ion2, temp=273.15, pres=101325.0)
        
        par[0]: time vector for particles, 1-D array, length n, unit days
        par[1]: diameter vector for particles, 1-D array, length m, unit nm
        par[2]: particle number-size distribution (dNdlogDp), 2-D array, n-by-m, unit cm-3 

        ion1[0]: time vector for the first ion polarity, 1-D array, days
        ion1[1]: diameter vector for the first ion polarity, 1-D array, nm
        ion1[2]: number-size distribution for the first ion polarity, 2-D array, cm-3

        ion2[0]: time vector for the second ion polarity, 1-D array, days
        ion2[1]: diameter vector for the second ion polarity, 1-D array, nm
        ion2[2]: number-size distribution for the second ion polarity, 2-D array, cm-3

        temp[:,0] = time vector associated with the temperatures, unit days
        temp[:,1] = temperatures in Kelvin
        also can be single float value, default 273.15 K

        pres[:,0] = time vector associated with the pressures, unit days
        pres[:,1] = pressure in Pascals
        also can be single float value, default 101325.0 Pa

        """
        
        # Close any existing figures
        plt.close('all')
        
        # Turn "ion mode" on
        self.particle_mode = 0
        self.ion_mode = 1
        
        self.par = par
        self.ion1 = ion1
        self.ion2 = ion2
        self.temp = temp
        self.pres = pres
        
        self.__init_ion_fig()
        self.__init_meteo()
        self.__init_variables()
        self.__init_polygons()
        self.__init_plots()
        self.__init_sliders()
        self.__init_buttons()
        self.__init_textboxes()
        
        plt.show()

    def __init_par_fig(self):
        
        # Figure and axes
        self.fig = plt.figure(figsize=(25,18))
        self.fig.subplots_adjust(left=0.1, bottom=0.25)

        self.ax1 = self.fig.add_subplot(121) # number-size distribution
        self.ax2 = self.fig.add_subplot(122) # formation rate
        
        self.ax1.set_ylabel("Diameter, [nm]")
        self.ax1.set_xlabel("Time, [days]")
        self.ax2.set_xlabel("Time, [days]")
        self.ax2.set_ylabel("Formation rate, [cm-3 s-1]")
        
        # Smoothing and color limits
        self.smooth = [0,0]
        self.clim = [1e1,1e5]
        self.dp_lim = [3,6]
        self.time_resolution = 1/1440.0
        
        # define time axis
        self.time_axis = np.arange(self.par[0][0] - np.floor(self.par[0][0]),self.par[0][-1] - np.floor(self.par[0][-1]), self.time_resolution)
        
        # Process the particle data
        self.par_df = pd.DataFrame(columns=self.par[1],index=self.par[0]-np.floor(self.par[0]),data=self.par[2])
        self.par_df = self.par_df.reindex(self.time_axis,method='nearest')
        self.par_time = self.par_df.index.values
        self.par_diam = self.par_df.columns.values
        self.par_data = self.par_df.values
        
        # Smoothed data
        self.smoothed_par_data = gaussian_filter(self.par_data,self.smooth,mode='constant')

        # Color plot
        mesh_par_dp, mesh_par_time = np.meshgrid(self.par_diam,self.par_time)
        self.pcplot = self.ax1.pcolormesh(mesh_par_time,mesh_par_dp,self.smoothed_par_data,\
                                          norm=colors.LogNorm(), linewidth=0, rasterized=True, cmap='jet',zorder=10)
        self.ax1.set_yscale('log')
        self.pcplot.set_clim(self.clim)
        self.pcplot.set_edgecolor('face')
        self.ax1.autoscale(tight='true')
        
    def __init_ion_fig(self):
        
        # Figure window and axes
        self.fig = plt.figure(figsize=(20,12))
        self.fig.subplots_adjust(left=0.1, bottom=0.25)
   
        self.ax1 = self.fig.add_subplot(121) # number-size distribution
        self.ax2 = self.fig.add_subplot(122) # formation rate
        
        self.ax1.set_ylabel("Diameter, [nm]")
        self.ax1.set_xlabel("Time, [days]")
        self.ax2.set_xlabel("Time, [days]")
        self.ax2.set_ylabel("Formation rate, [cm-3 s-1]")
        
        # Smoothing and color limits
        self.smooth = [0,0]
        self.clim = [1e1,1e4]
        self.dp_lim = [3,6]
        self.time_resolution = 1/1440.0
        
        # define time axis
        self.time_axis = np.arange(self.ion1[0][0]-np.floor(self.ion1[0][0]), self.ion1[0][-1]-np.floor(self.ion1[0][-1]), self.time_resolution)
        
        # Extract particle data
        self.par_df = pd.DataFrame(columns=self.par[1],index=self.par[0]-np.floor(self.par[0]),data=self.par[2])
        self.par_df = self.par_df.reindex(self.time_axis,method='nearest')
        self.par_time = self.par_df.index.values
        self.par_diam = self.par_df.columns.values
        self.par_data = self.par_df.values

        # the main ion data
        self.ion1_df = pd.DataFrame(columns=self.ion1[1],index=self.ion1[0]-np.floor(self.ion1[0]),data=self.ion1[2])
        self.ion1_df = self.ion1_df.reindex(self.time_axis,method='nearest')
        self.ion1_time = self.ion1_df.index.values
        self.ion1_diam = self.ion1_df.columns.values
        self.ion1_data = self.ion1_df.values    

        # auxiliary ion data
        self.ion2_df = pd.DataFrame(columns=self.ion2[1],index=self.ion2[0]-np.floor(self.ion2[0]),data=self.ion2[2])
        self.ion2_df = self.ion2_df.reindex(self.time_axis,method='nearest')
        self.ion2_time = self.ion2_df.index.values
        self.ion2_diam = self.ion2_df.columns.values
        self.ion2_data = self.ion2_df.values  
        
        # Smoothed data
        self.smoothed_ion1_data = gaussian_filter(self.ion1_data,self.smooth,mode='constant')
        self.smoothed_ion2_data = gaussian_filter(self.ion2_data,self.smooth,mode='constant')
        self.smoothed_par_data = gaussian_filter(self.par_data,self.smooth,mode='constant')

        # Color plot
        mesh_ion1_dp, mesh_ion1_time = np.meshgrid(self.ion1_diam,self.ion1_time)
        self.pcplot = self.ax1.pcolormesh(mesh_ion1_time,mesh_ion1_dp,self.smoothed_ion1_data,\
                                          norm=colors.LogNorm(), linewidth=0, rasterized=True, cmap='jet',zorder=10)
        self.ax1.set_yscale('log')
        self.pcplot.set_clim(self.clim)
        self.pcplot.set_edgecolor('face')
        self.ax1.autoscale(tight='true')
        

    def __init_meteo(self):
        
        # Single temperature value
        if isinstance(self.temp, float):
            self.temp_time = self.time_axis
            self.temp_data = self.temp*np.ones((len(self.temp_time),1))
        # Temperature time series
        else:
            self.temp_val = self.temp[:,1]
            self.temp_tim = self.temp[:,0] - np.floor(self.temp[:,0])
            self.temp_df = pd.DataFrame(index=self.temp_tim, data=self.temp_val)    
            self.temp_df = self.temp_df.reindex(self.time_axis,method='nearest')
            self.temp_data = self.temp_df.index.values
            self.temp_time = self.temp_df.values
            
        if isinstance(self.pres, float):
            self.pres_time = self.time_axis
            self.pres_data = self.pres*np.ones((len(self.pres_time),1))
        else:
            self.pres_val = self.pres[:,1]
            self.pres_tim = self.pres[:,0] - np.floor(self.pres[:,0]) 
            self.pres_df = pd.DataFrame(index=self.pres_tim, data=self.pres_val)
            self.pres_df = self.pres_df.reindex(self.time_axis,method='nearest')
            self.pres_data = self.pres_df.index.values
            self.pres_time = self.pres_df.values
          
    def __init_variables(self):
        self.CoagS = \
        self.J_time = \
        self.J = \
        self.J_lims = \
        self.mmd_time = \
        self.mmd_dp = \
        self.mmd_time_sr = \
        self.mmd_dp_sr = np.array([])
        self.J_peak = \
        self.gr = np.nan
        
    def __init_polygons(self):
        self.polyx = \
        self.polyy = \
        self.polyx_out = \
        self.polyy_out = np.array([])
        self.poly = Polygon(np.ones((2,2))*np.nan,True,facecolor='None',linewidth=1,edgecolor='k',zorder=3000)
        self.poly_out = Polygon(np.ones((2,2))*np.nan,True,facecolor='None',linewidth=1,edgecolor='r',zorder=5000) 
        self.par_poly_patch = self.ax1.add_patch(self.poly)        
        self.par_poly_out_patch = self.ax1.add_patch(self.poly_out)
        
    def __init_plots(self):
        self.mmd_plot = self.ax1.plot(np.nan,np.nan,'ko',zorder=5000)[0]
        self.mmd_plot_sr = self.ax1.plot(np.nan,np.nan,'mo',zorder=6000)[0]
        self.mmd_fit_sr = self.ax1.plot(np.nan,np.nan,'k-',zorder=8000)[0]
        self.J_plot = self.ax2.plot(np.nan,np.nan,'b-',zorder=5)[0]
        self.J_fit = self.ax2.plot(np.nan,np.nan,'k-',zorder=10)[0]
        
    def __init_sliders(self):
        self.color_axmin = self.fig.add_axes([0.1, 0.07, 0.05, 0.02])
        self.color_axmax = self.fig.add_axes([0.1, 0.1, 0.05, 0.02])
        self.smooth_axmin = self.fig.add_axes([0.1, 0.01, 0.05, 0.02])
        self.smooth_axmax = self.fig.add_axes([0.1, 0.04, 0.05, 0.02])
        self.color_smin = Slider(self.color_axmin,'color min',0,6,valinit=1)
        self.color_smax = Slider(self.color_axmax,'color max',0,6,valinit=5)
        self.smooth_smin = Slider(self.smooth_axmin,'x smooth',0,10,valinit=0)
        self.smooth_smax = Slider(self.smooth_axmax,'y smooth',0,10,valinit=0)
        self.color_smin.label.set_fontsize(self.fontsizes)
        self.color_smax.label.set_fontsize(self.fontsizes)
        self.smooth_smin.label.set_fontsize(self.fontsizes)
        self.smooth_smax.label.set_fontsize(self.fontsizes)
        self.color_smin.label.set_fontweight("bold")
        self.color_smax.label.set_fontweight("bold")
        self.smooth_smin.label.set_fontweight("bold")
        self.smooth_smax.label.set_fontweight("bold")
        self.color_smin.on_changed(self.__update_color)
        self.color_smax.on_changed(self.__update_color)
        self.smooth_smin.on_changed(self.__update_smooth)
        self.smooth_smax.on_changed(self.__update_smooth)
        
        self.reso_ax = self.fig.add_axes([0.1, 0.13, 0.05, 0.02])
        self.reso_s = Slider(self.reso_ax,'timereso',1.,60.,valinit=1.,valstep=1.)
        self.reso_s.label.set_fontsize(self.fontsizes)
        self.reso_s.label.set_fontweight("bold")
        self.reso_s.on_changed(self.__update_reso)
        
        # Do the size range selection using slidesr instead of text boxes
        self.dp_axmin = self.fig.add_axes([0.5, 0.01, 0.2, 0.03])
        self.dp_axmax = self.fig.add_axes([0.5, 0.05, 0.2, 0.03])
        self.dp_smin = Slider(self.dp_axmin,'dp min',1,20,valinit=self.dp_lim[0],valstep=0.5)
        self.dp_smax = Slider(self.dp_axmax,'dp max',1,20,valinit=self.dp_lim[1],valstep=0.5)
        self.dp_smin.label.set_fontsize(self.fontsizes)
        self.dp_smax.label.set_fontsize(self.fontsizes)
        self.dp_smin.label.set_fontweight("bold")
        self.dp_smax.label.set_fontweight("bold")
        self.dp_smin.on_changed(self.__update_dp)
        self.dp_smax.on_changed(self.__update_dp)
        
    def __init_buttons(self):
        
        # Area polygon
        self.poly_button_ax = self.fig.add_axes([0.21, 0.01, 0.1, 0.02])
        self.poly_button = Button(self.poly_button_ax, 'choose mode', color="white")
        self.poly_button.label.set_fontsize(self.fontsizes)
        self.poly_button.label.set_fontweight("bold")
        self.poly_button.on_clicked(self.__start_poly)
        self.poly_button_colors = itertools.cycle(['lime',"white"])
        
        # Outlier polygon
        self.poly_out_button_ax = self.fig.add_axes([0.21, 0.04, 0.1, 0.02])
        self.poly_out_button = Button(self.poly_out_button_ax, 'remove points',color="white")
        self.poly_out_button.label.set_fontsize(self.fontsizes)
        self.poly_out_button.label.set_fontweight("bold")
        self.poly_out_button.on_clicked(self.__start_poly_out)
        self.poly_out_button_colors = itertools.cycle(['lime',"white"])
        
        # Mode fitting button
        self.mode_fit_button_ax = self.fig.add_axes([0.35, 0.01, 0.1, 0.02])
        self.mode_fit_button = Button(self.mode_fit_button_ax,'mode fit',color="white")
        self.mode_fit_button.label.set_fontsize(self.fontsizes)
        self.mode_fit_button.label.set_fontweight("bold")
        self.mode_fit_button.on_clicked(self.__calc_mmd_modefit)

        # Maximum concentration fitting button
        self.maxconc_fit_button_ax = self.fig.add_axes([0.35, 0.04, 0.1, 0.02])
        self.maxconc_fit_button = Button(self.maxconc_fit_button_ax,'max conc',color="white")
        self.maxconc_fit_button.label.set_fontsize(self.fontsizes)
        self.maxconc_fit_button.label.set_fontweight("bold")
        self.maxconc_fit_button.on_clicked(self.__calc_mmd_maxconc) 
        
        # appearance time
        self.appearance_fit_button_ax = self.fig.add_axes([0.35, 0.07, 0.1, 0.02])
        self.appearance_fit_button = Button(self.appearance_fit_button_ax,'appearance',color="white")
        self.appearance_fit_button.label.set_fontsize(self.fontsizes)
        self.appearance_fit_button.label.set_fontweight("bold")
        self.appearance_fit_button.on_clicked(self.__calc_mmd_edge1) 

        # Upper edge
        self.upper_fit_button_ax = self.fig.add_axes([0.35, 0.1, 0.1, 0.02])
        self.upper_fit_button = Button(self.upper_fit_button_ax,'upper edge',color="white")
        self.upper_fit_button.label.set_fontsize(self.fontsizes)
        self.upper_fit_button.label.set_fontweight("bold")
        self.upper_fit_button.on_clicked(self.__calc_mmd_edge2) 

        # Particle formation rate / Ion formation rate
        self.fit_J_button_ax = self.fig.add_axes([0.21, 0.07, 0.1, 0.02])
        self.fit_J_button = Button(self.fit_J_button_ax, 'fit J', color="white")
        self.fit_J_button.label.set_fontsize(self.fontsizes)
        self.fit_J_button.label.set_fontweight("bold")
        self.fit_J_button_colors = itertools.cycle(['lime',"white"])
        self.J_bound_counter = 0
        self.fit_J_button.on_clicked(self.__start_J_fit)
        self.J_vertical_line1 = self.ax2.axvline(np.nan,c='k',ls='--')
        self.J_vertical_line2 = self.ax2.axvline(np.nan,c='k',ls='--')

        # Reset every value
        self.clear_all_button_ax = self.fig.add_axes([0.21, 0.1, 0.1, 0.02])
        self.clear_all_button = Button(self.clear_all_button_ax, 'clear all',color="white")
        self.clear_all_button.label.set_fontsize(self.fontsizes)
        self.clear_all_button.label.set_fontweight("bold")
        self.clear_all_button.on_clicked(self.__clear_all)

        # Clear the size range
        self.clear_sr_button_ax = self.fig.add_axes([0.21, 0.13, 0.1, 0.02])
        self.clear_sr_button = Button(self.clear_sr_button_ax, 'clear size-range',color="white")
        self.clear_sr_button.label.set_fontsize(self.fontsizes)
        self.clear_sr_button.label.set_fontweight("bold")
        self.clear_sr_button.on_clicked(self.__clear_sr)
        
    def __init_textboxes(self):

        self.box_gr_ax = self.fig.add_axes([0.9, 0.01, 0.05, 0.02])
        self.box_gr = TextBox(self.box_gr_ax,'GR (nm h-1)',initial = "%.2f" % self.gr)
        self.box_gr.label.set_fontsize(self.fontsizes)
        self.box_gr.label.set_fontweight("bold")
        
        self.box_J_peak_ax = self.fig.add_axes([0.9, 0.04, 0.05, 0.02])
        self.box_J_peak = TextBox(self.box_J_peak_ax,'J peak (cm-3 s-1)',initial = "%.2f" % self.J_peak)
        self.box_J_peak.label.set_fontsize(self.fontsizes)
        self.box_J_peak.label.set_fontweight("bold")
                      
        
    def __update_color(self,val):
        self.clim = [10**self.color_smin.val,10**self.color_smax.val]
        self.pcplot.set_clim(self.clim)
        self.fig.canvas.draw()
        
    def __update_dp(self,val):
        self.dp_lim = [self.dp_smin.val,self.dp_smax.val]
        self.J_bound_counter=0
        self.J_peak = np.nan
        self.J_fit.set_data(np.nan,np.nan)
        self.J_lims = np.array([])
        self.J_vertical_line1.set_xdata(np.nan)
        self.J_vertical_line2.set_xdata(np.nan)
        plt.draw()
        self.box_J_peak.set_val("%.2f" % self.J_peak) 
        self.__calc_gr()
        if self.particle_mode:
            self.__calc_J()
        if self.ion_mode:
            self.__calc_ion_J()
        
    def __update_smooth(self,val):
        self.smooth = [self.smooth_smin.val,self.smooth_smax.val]
        
        if self.particle_mode:
            self.smoothed_par_data = gaussian_filter(self.par_data,self.smooth,mode='constant')
            self.pcplot.set_array(self.smoothed_par_data[:-1,:-1].ravel())
        if self.ion_mode:
            self.smoothed_ion1_data = gaussian_filter(self.ion1_data,self.smooth,mode='constant')
            self.smoothed_ion2_data = gaussian_filter(self.ion2_data,self.smooth,mode='constant')
            self.smoothed_par_data = gaussian_filter(self.par_data,self.smooth,mode='constant')
            self.pcplot.set_array(self.smoothed_ion1_data[:-1,:-1].ravel())
        
        self.fig.canvas.draw()
        
    def __update_reso(self,val):
        
        self.time_resolution = self.reso_s.val/1440.
        
        if self.particle_mode:
            
            self.time_axis = np.arange(self.par[0][0] - np.floor(self.par[0][0]),self.par[0][-1] - np.floor(self.par[0][-1]), self.time_resolution)
            self.par_df = pd.DataFrame(columns=self.par[1],index=self.par[0]-np.floor(self.par[0]),data=self.par[2])
            self.par_df = self.par_df.reindex(self.time_axis,method='nearest')
            self.par_time = self.par_df.index.values
            self.par_diam = self.par_df.columns.values
            self.par_data = self.par_df.values
            
            # Smoothed data
            self.smoothed_par_data = gaussian_filter(self.par_data,self.smooth,mode='constant')
           
            # clean the pcolormesh plot
            self.pcplot.set_array(np.nan*self.smoothed_par_data[:-1,:-1].ravel())

            # Color plot
            mesh_par_dp, mesh_par_time = np.meshgrid(self.par_diam,self.par_time)
            self.pcplot = self.ax1.pcolormesh(mesh_par_time,mesh_par_dp,self.smoothed_par_data,\
                                              norm=colors.LogNorm(), linewidth=0, rasterized=True, cmap='jet',zorder=10)
            self.ax1.set_yscale('log')
            self.pcplot.set_clim(self.clim)
            self.pcplot.set_edgecolor('face')
            self.ax1.autoscale(tight='true')
            
        if self.ion_mode:
            
            # define time axis
            self.time_axis = np.arange(self.ion1[0][0]-np.floor(self.ion1[0][0]), self.ion1[0][-1]-np.floor(self.ion1[0][-1]), self.time_resolution)
            
            # Extract particle data
            self.par_df = pd.DataFrame(columns=self.par[1],index=self.par[0]-np.floor(self.par[0]),data=self.par[2])
            self.par_df = self.par_df.reindex(self.time_axis,method='nearest')
            self.par_time = self.par_df.index.values
            self.par_diam = self.par_df.columns.values
            self.par_data = self.par_df.values
    
            # the main ion data
            self.ion1_df = pd.DataFrame(columns=self.ion1[1],index=self.ion1[0]-np.floor(self.ion1[0]),data=self.ion1[2])
            self.ion1_df = self.ion1_df.reindex(self.time_axis,method='nearest')
            self.ion1_time = self.ion1_df.index.values
            self.ion1_diam = self.ion1_df.columns.values
            self.ion1_data = self.ion1_df.values    
    
            # auxiliary ion data
            self.ion2_df = pd.DataFrame(columns=self.ion2[1],index=self.ion2[0]-np.floor(self.ion2[0]),data=self.ion2[2])
            self.ion2_df = self.ion2_df.reindex(self.time_axis,method='nearest')
            self.ion2_time = self.ion2_df.index.values
            self.ion2_diam = self.ion2_df.columns.values
            self.ion2_data = self.ion2_df.values
            
            self.smoothed_ion1_data = gaussian_filter(self.ion1_data,self.smooth,mode='constant')
            self.smoothed_ion2_data = gaussian_filter(self.ion2_data,self.smooth,mode='constant')
            self.smoothed_par_data = gaussian_filter(self.par_data,self.smooth,mode='constant')
           
            self.pcplot.set_array(np.nan*self.smoothed_ion1_data[:-1,:-1].ravel())

            # Color plot
            mesh_ion1_dp, mesh_ion1_time = np.meshgrid(self.ion1_diam,self.ion1_time)
            self.pcplot = self.ax1.pcolormesh(mesh_ion1_time,mesh_ion1_dp,self.smoothed_ion1_data,\
                                              norm=colors.LogNorm(), linewidth=0, rasterized=True, cmap='jet',zorder=10)
            self.ax1.set_yscale('log')
            self.pcplot.set_clim(self.clim)
            self.pcplot.set_edgecolor('face')
            self.ax1.autoscale(tight='true') 
            
        self.fig.canvas.draw()
                            
    def __start_poly(self,event):
        button_color = next(self.poly_button_colors)
        if button_color=='lime':
            self.cid_poly = self.fig.canvas.mpl_connect('button_press_event',self.__draw_poly)
        elif button_color=='white':
            self.fig.canvas.mpl_disconnect(self.cid_poly)
        self.poly_button.color = button_color
        
    def __draw_poly(self,event):
        if event.inaxes==self.ax1:
            if event.button==1:
                self.polyx = np.append(self.polyx,event.xdata)
                self.polyy = np.append(self.polyy,event.ydata)
                self.poly.set_xy(np.array(list(zip(self.polyx,self.polyy))))
                plt.draw()
            elif event.button==3:
                self.polyx = np.array([])
                self.polyy = np.array([])
                self.poly.set_xy(np.ones((2,2))*np.nan)
                plt.draw()
                
    def __start_poly_out(self,event):
        button_color = next(self.poly_out_button_colors)
        if button_color=='lime':
            self.cid_poly_out = self.fig.canvas.mpl_connect('button_press_event',self.__draw_poly_out)
        elif button_color=='white':
            self.fig.canvas.mpl_disconnect(self.cid_poly_out)
        self.poly_out_button.color = button_color
        
    def __draw_poly_out(self,event):
        
        if event.inaxes == self.ax1:
            if event.button == 1:
                self.polyx_out = np.append(self.polyx_out,event.xdata)
                self.polyy_out = np.append(self.polyy_out,event.ydata)
                self.poly_out.set_xy(np.array(list(zip(self.polyx_out,self.polyy_out))))
                plt.draw()
            elif event.button==3:
                try:
                    perimeter_pts = Path(np.array(list(zip(self.polyx_out, self.polyy_out))))
                except ValueError:
                    return
                data_points = np.array(list(zip(self.mmd_time,self.mmd_dp)))
                if len(data_points)!=0:
                    boolean_mask = ~perimeter_pts.contains_points(data_points[:])
                    self.mmd_time = self.mmd_time[boolean_mask]
                    self.mmd_dp = self.mmd_dp[boolean_mask]
                    self.mmd_plot.set_data(self.mmd_time,self.mmd_dp)
                self.polyx_out = np.array([])
                self.polyy_out = np.array([])
                self.poly_out.set_xy(np.ones((2,2))*np.nan)
                plt.draw() 
        
    def __start_J_fit(self,event):
        button_color = next(self.fit_J_button_colors)
        if button_color=='lime':
            self.J_bound_counter=0
            self.J_peak = np.nan
            self.J_fit.set_data(np.nan,np.nan)
            self.J_lims = np.array([])
            self.J_vertical_line1.set_xdata(np.nan)
            self.J_vertical_line2.set_xdata(np.nan)
            plt.draw()
            self.box_J_peak.set_val("%.2f" % self.J_peak) 
            self.cid_fit_J = self.fig.canvas.mpl_connect('button_press_event',self.__fit_J)
        elif button_color=='white':
            self.fig.canvas.mpl_disconnect(self.cid_fit_J)
        self.fit_J_button.color = button_color
        

    def __fit_J(self,event):
        if event.inaxes==self.ax2:                        
            
            if event.button==1:
                if self.J_bound_counter==0:
                    self.J_lims = np.append(self.J_lims,event.xdata)
                    self.J_vertical_line1.set_xdata(event.xdata)
                    self.J_bound_counter = 1
                elif self.J_bound_counter==1:
                    self.J_lims = np.append(self.J_lims,event.xdata)
                    self.J_vertical_line2.set_xdata(event.xdata)
                    self.J_bound_counter = 2
                    findex = np.argwhere((self.J_time<=self.J_lims.max()) & 
                                        (self.J_time>=self.J_lims.min()))\
                                       .flatten()
                    x = self.J_time[findex]
                    y = self.J[findex]
                    mu = np.nanmean(x)
                    a = np.max(y)
                    sigma = np.nanstd(x)
                    try:
                        params,pcov = curve_fit(self.__gaus,x,y,p0=[a,mu,sigma])
    
                        # Amplitude of the peak is J
                        self.J_peak = params[0]
                       
                        # Plot the fit for visual verification
                        fit = self.__gaus(self.J_time,params[0],params[1],params[2])
                        self.J_fit.set_data(self.J_time,fit)
                        self.ax2.set_ylim(bottom=np.nanmin(np.append(self.J,0)))
    
                    except:
                        print ("Diverges")
    
                elif self.J_bound_counter==2:
                    pass
                
                plt.draw()
                self.box_J_peak.set_val("%.2f" % self.J_peak) 
                
            if event.button==3:
                self.J_bound_counter=0
                self.J_peak = np.nan
                self.J_fit.set_data(np.nan,np.nan)
                self.J_lims = np.array([])
                self.J_vertical_line1.set_xdata(np.nan)
                self.J_vertical_line2.set_xdata(np.nan)
                plt.draw()
                self.box_J_peak.set_val("%.2f" % self.J_peak)         
        
        

    def __logi(self,x,L,x0,k):
        return L * (1 + np.exp(-k*(x-x0)))**(-1) 
        
    def __gaus(self,x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
                
    def __calc_mmd_modefit(self,event):
        """ Calculate mean mode diameters """
        
        # Use smoothed data
        if self.particle_mode:
            data = np.log10(gaussian_filter(self.par_data,self.smooth,mode='constant'))
            dpdp,tt = np.meshgrid(self.par_diam,self.par_time)
            points = np.concatenate((tt.flatten()[np.newaxis].T,
                                     dpdp.flatten()[np.newaxis].T,
                                     data.flatten()[np.newaxis].T),
                                     axis=1)
        if self.ion_mode:
            data = np.log10(gaussian_filter(self.ion1_data,self.smooth,mode='constant'))
            dpdp,tt = np.meshgrid(self.ion1_diam,self.ion1_time)
            points = np.concatenate((tt.flatten()[np.newaxis].T,
                                     dpdp.flatten()[np.newaxis].T,
                                     data.flatten()[np.newaxis].T),
                                     axis=1)

        # Transform polygon perimeter to path
        try:
            banana_perimeter = Path(np.array(list(zip(self.polyx,self.polyy))))
        except ValueError:
            print ("No polygon found")
            return

        # Eliminate nans and infs from dndlogdp
        points = np.delete(points,np.argwhere((np.isnan(points[:,2]))|(np.isinf(points[:,2]))),axis=0)
        banana_points = points[banana_perimeter.contains_points(points[:,[0,1]]),:]

        if len(banana_points)==0:
            print ("Found no points inside polygon.")
            return
        
        # Grouping the size distribution data points
        if self.particle_mode:
            pre_sorted_banana_points = [banana_points[banana_points[:,0]==x,:] for x in self.par_time if x in banana_points[:,0]]
        if self.ion_mode:
            pre_sorted_banana_points = [banana_points[banana_points[:,0]==x,:] for x in self.ion1_time if x in banana_points[:,0]]
            
        sorted_banana_points = [x[x[:,1].argsort()] for x in pre_sorted_banana_points]

        for i in range(0,len(sorted_banana_points)):
            x = np.log10(sorted_banana_points[i][:,1])
            y = sorted_banana_points[i][:,2]
            a = np.max(y)
            mu = np.mean(x)
            sigma = np.std(x)
            try:
                params,pcov = curve_fit(self.__gaus,x,y,p0=[a,mu,sigma])
                print(params[1])
                if ((params[1]>=x.max()) | (params[1]<=x.min())):
                    print ("Peak outside range. Skipping %f" % (sorted_banana_points[i][0,0]))
                else:
                    self.mmd_time = np.append(self.mmd_time,sorted_banana_points[i][0,0])
                    self.mmd_dp = np.append(self.mmd_dp,10**params[1])
            except:
                print ("Diverges. Skipping %f" % (sorted_banana_points[i][0,0]))

        # Plot the result on ax
        self.mmd_plot.set_data(self.mmd_time,self.mmd_dp)
        plt.draw()
        
    def __calc_mmd_maxconc(self,event):
        """ Calculate mean mode diameters
        """
        
        # Use smoothed data
        if self.particle_mode:
            data = np.log10(gaussian_filter(self.par_data,self.smooth,mode='constant'))
            dpdp,tt = np.meshgrid(self.par_diam,self.par_time)
            points = np.concatenate((tt.flatten()[np.newaxis].T,
                                     dpdp.flatten()[np.newaxis].T,
                                     data.flatten()[np.newaxis].T),
                                     axis=1)
        if self.ion_mode:
            data = np.log10(gaussian_filter(self.ion1_data,self.smooth,mode='constant'))
            dpdp,tt = np.meshgrid(self.ion1_diam,self.ion1_time)
            points = np.concatenate((tt.flatten()[np.newaxis].T,
                                     dpdp.flatten()[np.newaxis].T,
                                     data.flatten()[np.newaxis].T),
                                     axis=1)

        # Transform polygon perimeter to path
        try:
            banana_perimeter = Path(np.array(list(zip(self.polyx,self.polyy))))
        except ValueError:
            print ("No polygon found")
            return

        # Eliminate nans and infs from dndlogdp
        points = np.delete(points,np.argwhere((np.isnan(points[:,2]))|(np.isinf(points[:,2]))),axis=0)
        banana_points = points[banana_perimeter.contains_points(points[:,[0,1]]),:]

        if len(banana_points)==0:
            print ("Found no points inside polygon.")
            return
        
        # Grouping the size distribution data points
        if self.particle_mode:
            pre_sorted_banana_points = [banana_points[banana_points[:,1]==x,:] for x in self.par_diam if x in banana_points[:,1]]
        if self.ion_mode:
            pre_sorted_banana_points = [banana_points[banana_points[:,1]==x,:] for x in self.ion1_diam if x in banana_points[:,1]]
            
        sorted_banana_points = [x[x[:,0].argsort()] for x in pre_sorted_banana_points]
        
        for i in range(0,len(sorted_banana_points)):
            x = sorted_banana_points[i][:,0]
            y = sorted_banana_points[i][:,2]
            a=np.max(y)
            mu=np.mean(x)
            sigma=np.std(x)
            try:
                params,pcov = curve_fit(self.__gaus,x,y,p0=[a,mu,sigma])
                if ((params[1]>=x.max()) | (params[1]<=x.min())):
                    print ("Peak outside range. Skipping %f" % (sorted_banana_points[i][0,1]))
                else:
                    self.mmd_dp = np.append(self.mmd_dp,sorted_banana_points[i][0,1])
                    self.mmd_time = np.append(self.mmd_time,params[1])
            except:
                print ("Diverges. Skipping %f" % (sorted_banana_points[i][0,1]))

        # Plot the result on ax
        self.mmd_plot.set_data(self.mmd_time,self.mmd_dp)
        plt.draw()
        
        
    def __calc_mmd_edge1(self,event):
        """ Calculate mean mode diameters
        """
        
        # Use smoothed data
        if self.particle_mode:
            data = gaussian_filter(self.par_data,self.smooth,mode='constant')
            dpdp,tt = np.meshgrid(self.par_diam,self.par_time)
            points = np.concatenate((tt.flatten()[np.newaxis].T,
                                     dpdp.flatten()[np.newaxis].T,
                                     data.flatten()[np.newaxis].T),
                                     axis=1)
        if self.ion_mode:
            data = np.log10(gaussian_filter(self.ion1_data,self.smooth,mode='constant'))
            dpdp,tt = np.meshgrid(self.ion1_diam,self.ion1_time)
            points = np.concatenate((tt.flatten()[np.newaxis].T,
                                     dpdp.flatten()[np.newaxis].T,
                                     data.flatten()[np.newaxis].T),
                                     axis=1)

        # Transform polygon perimeter to path
        try:
            banana_perimeter = Path(np.array(list(zip(self.polyx,self.polyy))))
        except ValueError:
            print ("No polygon found")
            return

        # Eliminate nans and infs from dndlogdp
        points = np.delete(points,np.argwhere((np.isnan(points[:,2]))|(np.isinf(points[:,2]))),axis=0)
        banana_points = points[banana_perimeter.contains_points(points[:,[0,1]]),:]

        if len(banana_points)==0:
            print ("Found no points inside polygon.")
            return
        
        # Grouping the size distribution data points
        if self.particle_mode:
            pre_sorted_banana_points = [banana_points[banana_points[:,1]==x,:] for x in self.par_diam if x in banana_points[:,1]]
        if self.ion_mode:
            pre_sorted_banana_points = [banana_points[banana_points[:,1]==x,:] for x in self.ion1_diam if x in banana_points[:,1]]
            
        sorted_banana_points = [x[x[:,0].argsort()] for x in pre_sorted_banana_points]
        
        for i in range(0,len(sorted_banana_points)):
            x = sorted_banana_points[i][:,0]
            y = sorted_banana_points[i][:,2] - np.min(sorted_banana_points[i][:,2])
            L = np.max(y)
            x0 = np.nanmean(x)
            k = 1.0
            try:
                params,pcov = curve_fit(self.__logi,x,y,p0=[L,x0,k])
                if ((params[1]>=x.max()) | (params[1]<=x.min())):
                    print ("Peak outside range. Skipping %f" % (sorted_banana_points[i][0,1]))
                else:
                    self.mmd_dp = np.append(self.mmd_dp,sorted_banana_points[i][0,1])
                    self.mmd_time = np.append(self.mmd_time,params[1])
            except:
                print ("Diverges. Skipping %f" % (sorted_banana_points[i][0,1]))

        # Plot the result on ax
        self.mmd_plot.set_data(self.mmd_time,self.mmd_dp)
        plt.draw()
        
        
    def __calc_mmd_edge2(self,event):
        """ Calculate mean mode diameters
        """
        
        # Use smoothed data
        if self.particle_mode:
            data = gaussian_filter(self.par_data,self.smooth,mode='constant')
            dpdp,tt = np.meshgrid(self.par_diam,self.par_time)
            points = np.concatenate((tt.flatten()[np.newaxis].T,
                                     dpdp.flatten()[np.newaxis].T,
                                     data.flatten()[np.newaxis].T),
                                     axis=1)
        if self.ion_mode:
            data = np.log10(gaussian_filter(self.ion1_data,self.smooth,mode='constant'))
            dpdp,tt = np.meshgrid(self.ion1_diam,self.ion1_time)
            points = np.concatenate((tt.flatten()[np.newaxis].T,
                                     dpdp.flatten()[np.newaxis].T,
                                     data.flatten()[np.newaxis].T),
                                     axis=1)

        # Transform polygon perimeter to path
        try:
            banana_perimeter = Path(np.array(list(zip(self.polyx,self.polyy))))
        except ValueError:
            print ("No polygon found")
            return

        # Eliminate nans and infs from dndlogdp
        points = np.delete(points,np.argwhere((np.isnan(points[:,2]))|(np.isinf(points[:,2]))),axis=0)
        banana_points = points[banana_perimeter.contains_points(points[:,[0,1]]),:]

        if len(banana_points)==0:
            print ("Found no points inside polygon.")
            return
        
        # Grouping the size distribution data points
        if self.particle_mode:
            pre_sorted_banana_points = [banana_points[banana_points[:,0]==x,:] for x in self.par_time if x in banana_points[:,0]]
        if self.ion_mode:
            pre_sorted_banana_points = [banana_points[banana_points[:,0]==x,:] for x in self.ion1_time if x in banana_points[:,0]]
            
        sorted_banana_points = [x[x[:,1].argsort()] for x in pre_sorted_banana_points]

        for i in range(0,len(sorted_banana_points)):
            x = np.log10(sorted_banana_points[i][:,1])
            y = sorted_banana_points[i][:,2] - np.min(sorted_banana_points[i][:,2])
            L = np.max(y)
            x0 = np.nanmean(x)
            k = 1
            try:
                params,pcov = curve_fit(self.__logi,x,y,p0=[L,x0,k])
                print(params[1])
                if ((params[1]>=x.max()) | (params[1]<=x.min())):
                    print ("Peak outside range. Skipping %f" % (sorted_banana_points[i][0,0]))
                else:
                    self.mmd_time = np.append(self.mmd_time,sorted_banana_points[i][0,0])
                    self.mmd_dp = np.append(self.mmd_dp,10**params[1])
            except:
                print ("Diverges. Skipping %f" % (sorted_banana_points[i][0,0]))

        # Plot the result on ax
        self.mmd_plot.set_data(self.mmd_time,self.mmd_dp)
        plt.draw()      
        
        
        
    def __calc_gr(self):
        """ Calculate GR
        """
        
        findex = np.argwhere((self.mmd_dp>=self.dp_lim[0]) &
                             (self.mmd_dp<=self.dp_lim[1])).flatten()

        if len(findex)<=1:
            print ("Less than or equal to one point")
            return

        self.mmd_time_sr = self.mmd_time[findex]
        self.mmd_dp_sr = self.mmd_dp[findex]

        # Fit a line to the chosen points
        params = np.polyfit(self.mmd_time_sr,self.mmd_dp_sr,1)
        self.gr = params[0]/24. # nm h-1

        # Show the points in the given size-range as magenta color
        self.mmd_plot_sr.set_data(self.mmd_time_sr,self.mmd_dp_sr)

        # Draw the line fit to the points also
        x = self.mmd_time_sr
        y = params[0]*x + params[1]
        self.mmd_fit_sr.set_data(x,y)
        plt.draw()
        
        self.box_gr.set_val("%.2f" % self.gr)

    def __dNdlog2dN(self,Dp,dNdlogDp):
        """ Convert from DnDlogDp to dN """
          
        x = np.log10(Dp)
        y = (x[1:]+x[:-1])/2.
        y = np.pad(y,1,'constant',constant_values=(x[0]-(y[0]-x[0]),x[-1]+(x[-1]-y[-1])))
        dlogDp = np.diff(y)
        return dNdlogDp*dlogDp # cm-3

    def __calc_concentration(self, diam, data, dmin, dmax):
        """ Calculate number concentration between Dp1 and Dp2
        data is the size distribution
        diam is the diameters """

        dp = np.log10(diam*1e-9)
        conc = data # smoothed
        dmin = np.max((np.log10(dmin),dp[0]))
        dmax = np.min((np.log10(dmax),dp[-1]))
        dpi = np.arange(dmin,dmax,0.001)
        conci = np.sum(interp1d(dp,conc,kind='nearest')(dpi)*0.001,axis=1)
        return conci
    
    def __calc_CoagS(self):
        """ Calculate CoagS, also accounting for temperature and pressure changes """

        Dp_small = self.dp_lim[0]*1e-9 # in m
        temp = self.temp_data # Kelvin
        pres = self.pres_data # Pascal
        Dp = self.par_diam*1e-9 # m
        time = self.par_time # days
        N = self.__dNdlog2dN(Dp,self.smoothed_par_data) # cm-3
        findex = np.argwhere(Dp>=Dp_small).flatten()
        big_R = Dp[findex]/2.
        big_N = N[:,findex]
        k_B = 1.38064852e-23 # Boltzmann constant m2 kg s-2 K-1
        r0=Dp_small/2.
        r1=r0
        dens=1000.
        self.CoagS=np.zeros(time.shape)
        for i in range(0,len(time)):
            lamda=(6.73e-8*temp[i]*(1+(110.4/temp[i])))/(296*pres[i]/101325.0*1.373)
            myy=(1.832e-5*(temp[i]**(1.5))*406.4)/(5093*(temp[i]+110.4))
            kn1=lamda/r1
            kn=lamda/big_R
            CC= 1.+(kn*(1.142+(0.558*np.exp((-.999)/kn))))
            CC1= 1. + (kn1*(1.142+(0.558*np.exp((-.999)/kn1))))
            D = (k_B*temp[i]*CC)/(6.*np.pi*myy*big_R)
            D1 = (k_B*temp[i]*CC1)/(6.*np.pi*myy*r1)
            M = 4./3.*np.pi*(big_R**3)*dens
            M1 = 4./3.*np.pi*(r1**3)*dens
            c= np.sqrt((8.*k_B*temp[i])/(np.pi*M))
            c1= np.sqrt((8.*k_B*temp[i])/(np.pi*M1))
            c12= np.sqrt((c**2)+(c1**2))
            r12= big_R+r1
            D12= D+D1
            CCONT = 4.*np.pi*r12*D12
            CFR = np.pi*r12*r12*c12
            L=(8.*D)/(np.pi*c)
            L1=(8.*D1)/(np.pi*c1)
            SIG=(1./(3.*r12*L))*((r12+L)**3-(r12*r12+L*L)**1.5)-r12
            SIG1=(1./(3.*r12*L1))*((r12+L1)**3-(r12*r12+L1*L1)**1.5)-r12
            SIG12= np.sqrt((SIG**2)+(SIG1**2))
            KO=CCONT/((r12/(r12+SIG12))+(CCONT/CFR))
            self.CoagS[i] = np.nansum(KO*big_N[i,:]*1e6)
            if (r0==big_R[0]):
                self.CoagS[i] = 0.5*KO*big_N[i,0]*1e6+np.nansum(KO*big_N[i,1:]*1e6)
            else:
                self.CoagS[i] = np.nansum(KO*big_N[i,:]*1e6)


    def __calc_J(self):
        """ Calculate J """
        
        if np.isnan(self.gr):
            return

        self.__calc_CoagS() # s-1
        dp1 = self.dp_lim[0]*1e-9 # meters
        dp2 = self.dp_lim[1]*1e-9 # meters
        time = self.par_time*1.157e5 # days -> seconds
        N = self.__calc_concentration(self.par_diam,self.smoothed_par_data,dp1,dp2) # cm-3
        GR = 2.778e-13*self.gr # nm/h -> m/s
        mid_time = (time[1:] + time[:-1])/2.0/1.157e5 # s -> days
        dNdt = np.diff(N)/np.diff(time) # derivative cm-3 s-1
        mid_N = (N[1:] + N[:-1])/2.0
        mid_CoagS = (self.CoagS[1:] + self.CoagS[:-1])/2.0
        mid_J = dNdt + mid_CoagS * mid_N + GR/(dp2-dp1) * mid_N

        self.J_time = mid_time
        self.J = mid_J

        self.J_plot.set_data(self.J_time,self.J)
        self.ax2.autoscale(tight=1)

        self.ax2.set_xlim((self.J_time.min(),self.J_time.max()))
        self.ax2.set_ylim((np.nanmin(np.append(self.J,0)),np.nanmax(np.append(self.J,self.J_peak))))
        plt.draw()
        
    def __calc_ion_J(self):
        """ Calculate J for ion polarity 1"""
        
        if np.isnan(self.gr):
            return

        self.__calc_CoagS() # s-1
        
        dp1 = self.dp_lim[0]*1e-9 # meters
        dp2 = self.dp_lim[1]*1e-9 # meters
        
        # Time axis is the same for every data
        time = self.ion1_time*1.157e5 # days -> seconds
        
        par_N = self.__calc_concentration(self.par_diam,self.smoothed_par_data,dp1,dp2) # cm-3
        ion1_N = self.__calc_concentration(self.ion1_diam,self.smoothed_ion1_data,dp1,dp2) # cm-3
        ion1_lessN = self.__calc_concentration(self.ion1_diam,self.smoothed_ion1_data,1e-10,dp2) # cm-3
        ion2_lessN = self.__calc_concentration(self.ion2_diam,self.smoothed_ion2_data,1e-10,dp2) # cm-3
        
        GR = 2.778e-13*self.gr # nm/h -> m/s
        mid_time = (time[1:] + time[:-1])/2.0/1.157e5 # s -> days
        ion1_dNdt = np.diff(ion1_N)/np.diff(time) # derivative cm-3 s-1
        
        par_mid_N = (par_N[1:] + par_N[:-1])/2.0
        ion1_mid_N = (ion1_N[1:] + ion1_N[:-1])/2.0
        ion1_mid_lessN = (ion1_lessN[1:] + ion1_lessN[:-1])/2.0
        ion2_mid_lessN = (ion2_lessN[1:] + ion2_lessN[:-1])/2.0
        
        mid_CoagS = (self.CoagS[1:] + self.CoagS[:-1])/2.0
        
        alpha = 1.6e-6 # cm3 s-1
        Xi = 0.01e-6 # cm3 s-1
        
        mid_J = ion1_dNdt\
                + mid_CoagS * ion1_mid_N\
                + GR/(dp2-dp1) * ion1_mid_N\
                + alpha * ion1_mid_N * ion2_mid_lessN\
                - Xi * par_mid_N * ion1_mid_lessN 

        self.J_time = mid_time
        self.J = mid_J

        self.J_plot.set_data(self.J_time,self.J)
        self.ax2.set_xlim((self.J_time.min(),self.J_time.max()))
        self.ax2.set_ylim((np.nanmin(np.append(self.J,0)),np.nanmax(np.append(self.J,self.J_peak))))
        plt.draw()
        
        

    def __clear_sr(self,event):
        """ Clear points in the size-range and the fit to them """
        
        self.CoagS = \
        self.J_time = \
        self.J = \
        self.J_lims = \
        self.mmd_time_sr = \
        self.mmd_dp_sr = np.array([])
        
        self.J_peak = \
        self.gr = np.nan

        self.mmd_plot_sr.set_data(np.nan,np.nan)
        self.mmd_fit_sr.set_data(np.nan,np.nan)
        self.J_plot.set_data(np.nan,np.nan)
        self.J_fit.set_data(np.nan,np.nan)
        self.J_vertical_line1.set_xdata(np.nan)
        self.J_vertical_line2.set_xdata(np.nan)
        
        plt.draw()
        
        self.box_gr.set_val("%.2f" % self.gr)
        self.box_J_peak.set_val("%.2f" % self.J_peak)

    def __clear_all(self,event):
        """ Clear polygons and average mode diameters """
        
        self.CoagS = \
        self.J_time = \
        self.J = \
        self.J_lims = \
        self.mmd_time = \
        self.mmd_dp = \
        self.mmd_time_sr = \
        self.mmd_dp_sr = np.array([])
        
        # Initialize all np.nan variables
        self.J_peak = \
        self.gr = np.nan
        
        # Clears polygon used to outline particle mode
        self.polyx = \
        self.polyy = \
        self.polyx_out = \
        self.polyy_out = np.array([])
        self.poly.set_xy(np.ones((2,2))*np.nan)
        self.poly_out.set_xy(np.ones((2,2))*np.nan)
        
        self.box_gr.set_val("%.2f" % self.gr)
        self.box_J_peak.set_val("%.2f" % self.J_peak)
        
        # clear average mode diameters and fit
        self.mmd_plot.set_data(np.nan,np.nan)
        self.mmd_plot_sr.set_data(np.nan,np.nan)
        self.mmd_fit_sr.set_data(np.nan,np.nan)
        self.J_plot.set_data(np.nan,np.nan)
        self.J_fit.set_data(np.nan,np.nan)
        self.J_vertical_line1.set_xdata(np.nan)
        self.J_vertical_line2.set_xdata(np.nan)
        plt.draw()


