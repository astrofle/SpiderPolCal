"""

"""

import numpy as np

from spiderpolcal import config


class Raw2Stokes():
    """
    """

    def __init__(self, freq, table, c2K_xx, c2K_yy, tht_off, tht_slp, tht_obs,
                 do_amp_cal=False):
        """
        freq : array
            Frequency axis in Hz.
        table : recarray
            SDFITS subset with the data to be calibrated.
            Assumes this is one Spider leg.
        c2K_xx : array
            Count to Kelvin conversion factor for XX product.
        c2K_yy : array
            Count to Kelvin conversion factor for YY product.
        tht_off : array
            Phase zero offset.
        tht_slp : array
            Phase slope.
        """

        self.freq = freq
        self.table = table
        self.c2K_xx = c2K_xx
        self.c2K_yy = c2K_yy
        self.tht_off = tht_off
        self.tht_slp = tht_slp
        self.tht_obs = tht_obs

        self.calibrate(do_amp_cal)
        self.to_Stokes()


    def split_pol(self):
        """
        """

        def table_mask(pol):
            """
            pol : {"XX", "YY", "XY", "YX"}
                Polarization product.
            """
            return (self.table["PLNUM"] == config.pol2indx[pol])

        # XX noise diode ON.
        mask = table_mask("XX")
        self.table_xx = self.table[mask]
        # YY, XY and YX.
        self.table_yy = self.table[table_mask("YY")]
        self.table_xy = self.table[table_mask("XY")]
        self.table_yx = self.table[table_mask("YX")]
 

    def split_sig_ref(self):
        """
        """

        def split(table, int_idx, beam_width, peak_width=2):
            """
            table : recarray
                
            int_idx : int
                Integration with the peak amplitude (on source).
            beam_width : int
                Number of integrations around the peak to ignore.
                The off is selected as int_idx-beam_width and 
                int_idx+beam_width.
            peak_width : int
                Number of integration to include in the on source
                portion.
            """
            w = peak_width//2
            return {"on": table[40:41],
                    "off": np.concatenate((table[5:14], 
                                           table[65:74]))
                   }
#            return {"on": table[int_idx-w:int_idx+w+1],
#                    "off": np.concatenate((table[:int_idx-beam_width], 
#                                           table[int_idx+beam_width:]))
#                   }
        
        self.xx = split(self.table_xx, 40, 15)
        self.yy = split(self.table_yy, 40, 15)
        self.xy = split(self.table_xy, 40, 15)
        self.yx = split(self.table_yx, 40, 15)


    def time_average(self, data):
        """
        """
        
        weights = data["EXPOSURE"]/np.power(data["TSYS"], 2)
        
        return np.average(data["DATA"], axis=0, weights=weights)
        

    def average_refs(self):
        """
        """

        self.xx_ref = self.time_average(self.xx["off"])
        self.yy_ref = self.time_average(self.yy["off"])
        self.xy_ref = self.time_average(self.xy["off"])
        self.yx_ref = self.time_average(self.yx["off"])


    def average_sigs(self):
        """
        """

        self.xx_sig = self.time_average(self.xx["on"])
        self.yy_sig = self.time_average(self.yy["on"])
        self.xy_sig = self.time_average(self.xy["on"])
        self.yx_sig = self.time_average(self.yx["on"])


    def calibrate(self, do_amp_cal):
        """
        """

        self.split_pol()
        self.split_sig_ref()
        self.average_sigs()
        self.average_refs()
        
        # Calibrate the raw counts to Kelvin, 
        # removing the bandpass shape.
        self.xx_r2c = Raw2CalAuto(self.freq, self.xx_sig, self.xx_ref, self.c2K_xx, 
                                  do_amp_cal=do_amp_cal)
        self.xx_cal = self.xx_r2c.cal
        self.yy_r2c = Raw2CalAuto(self.freq, self.yy_sig, self.yy_ref, self.c2K_yy,
                                  do_amp_cal=do_amp_cal)
        self.yy_cal = self.yy_r2c.cal
        self.xy_r2c = Raw2CalCross(self.freq, self.xy_sig, self.xy_ref, 
                                   self.c2K_xx, self.c2K_yy, 
                                   self.xx_ref, self.yy_ref,
                                   do_amp_cal=do_amp_cal,
                                   )
        xy_cal = self.xy_r2c.cal
        self.yx_r2c = Raw2CalCross(self.freq, self.yx_sig, self.yx_ref, 
                                   self.c2K_xx, self.c2K_yy, 
                                   self.xx_ref, self.yy_ref,
                                   do_amp_cal=do_amp_cal,
                                   )
        yx_cal = self.yx_r2c.cal
        
        # Phase calibration for cross-correlation products.
        self.phase_cal = crossPhaseCal(self.freq, xy_cal, yx_cal, 
                                       self.tht_off, self.tht_slp,
                                       self.tht_obs)
        self.xy_cal = self.phase_cal.xy_cal
        self.yx_cal = self.phase_cal.yx_cal

    
    def to_Stokes(self):
        """
        """

        self.i_obs = self.xx_cal + self.yy_cal
        self.q_obs = self.xx_cal - self.yy_cal
        self.u_obs = 2.*self.xy_cal
        self.v_obs = 2.*self.yx_cal


class crossPhaseCal():
    """
    """

    def __init__(self, freq, xy, yx, tht_off, tht_slp, tht_obs):
        """
        """

        self.freq = freq
        self.xy = xy
        self.yx = yx
        self.tht_off = tht_off
        self.tht_slp = tht_slp
        self.tht_obs = tht_obs
        self.tht_fit = self.tht_off + self.tht_slp*self.freq.to("MHz").value
        self.phase_cal()
   

    def phase_cal(self):
        """
        """

        rot = (self.xy + 1j*self.yx)*np.exp(-1j*(self.tht_obs))
        
        self.xy_cal = rot.real
        self.yx_cal = rot.imag 


class Raw2CalCross():
    """
    """

    def __init__(self, freq, sig, ref, c2K_xx, c2K_yy, 
                 bandpass_xx, bandpass_yy, do_amp_cal=False):
        """
        """

        self.freq = freq
        self.sig = sig
        self.ref = ref
        self.c2K = None
        self.compute_c2K(c2K_xx, c2K_yy)
        self.bandpass = None
        self.compute_bandpass(bandpass_xx, bandpass_yy)
        # Calibrate the data.
        self.set_scale_factor(do_amp_cal)
        self.cal = None
        self.amp_cal()


    def compute_c2K(self, xx, yy):
        """
        """

        self.c2K = np.sqrt( xx*yy )


    def compute_bandpass(self, xx, yy):
        """
        """

        self.bandpass = np.sqrt( xx*yy )

    
    def set_scale_factor(self, do_amp_cal):
        """
        """
        if do_amp_cal:
            self.scale = np.nanmean(self.bandpass)
        else:
            self.scale = 1./np.nanmean(self.c2K)


    def amp_cal(self):
        """
        """

        self.cal = self.c2K*(self.sig - self.ref)/self.bandpass * np.mean(self.scale)


class Raw2CalAuto():
    """
    """

    def __init__(self, freq, sig, ref, c2K, do_amp_cal=False):
        """
        """

        self.sig = sig
        self.ref = ref
        self.c2K = c2K
        # Calibrate.
        self.set_scale_factor(do_amp_cal)
        self.cal = None
        self.amp_cal()


    def amp_cal(self):
        """
        """

        self.cal = self.c2K*(self.sig - self.ref)/self.ref * np.mean(self.scale)


    def set_scale_factor(self, do_amp_cal):
        """
        """

        if do_amp_cal:
            self.scale = np.nanmean(self.ref)
        else:
            self.scale = 1./np.nanmean(self.c2K)


class DataCalInitCross():
    """
    """

    def __init__(self, freq, data_xy_on, data_xy_off, data_yx_on, data_yx_off):
        """
        """

        self.freq = freq
        self.data_xy_on = data_xy_on
        self.data_xy_off = data_xy_off
        self.data_yx_on = data_yx_on
        self.data_yx_off = data_yx_off

        self.theta_obs = None
        self.theta_fit = None
        self.theta_off = None
        self.theta_slp = None
    

    def phase_fit(self, order=1):
        """
        """

        data_xy = self.data_xy_on - self.data_xy_off
        data_yx = self.data_yx_on - self.data_yx_off

        # Angles in radians.
        self.theta_obs = np.arctan2(data_yx, data_xy)

        x = (self.freq - np.median(self.freq)).to("MHz").value

        self.theta_fit = np.empty(self.theta_obs.shape, dtype=float)
        self.theta_off = np.empty(len(self.theta_obs), dtype=float)
        self.theta_slp = np.empty(len(self.theta_obs), dtype=float)

        for i in range(len(self.theta_obs)):
            self.pfit = np.polyfit(x, self.theta_obs[i], order)
            self.theta_fit[i] = np.poly1d(self.pfit)(x)

            self.theta_off[i] = self.pfit[order]
            self.theta_slp[i] = self.pfit[order-1]


class DataCalInitAuto():
    """
    """

    def __init__(self, freq, data_on, data_off, tcal):

        self.freq = freq
        self.data_on = data_on
        self.data_off = data_off
        self.tcal = tcal

        self.c2K = None
        self.bandpass = None


    def compute_c2K(self, verbose=False):
        """
        Computes the counts to Kelvin conversion factor.
        """

        self.c2K = self.tcal/self.bandpass


    def compute_bandpass(self):
        """
        """

        self.bandpass = (self.data_on - self.data_off)

    
    def compute_tsys(self):
        """
        """

        self.tsys = self.data_off*self.c2K


    def cal_init(self):
        """
        """
        self.compute_bandpass()
        self.compute_c2K()
        self.compute_tsys()


class DataCal():


    def __init__(self, freq, table, tcal_xx=None, tcal_yy=None):
        """
        """
        self.freq = freq
        self.table = table

        self.data_xx_on = None
        self.data_yy_on = None
        self.data_xy_on = None
        self.data_yx_on = None
        self.data_xx_off = None 
        self.data_yy_off = None
        self.data_xy_off = None
        self.data_yx_off = None
        self.tcal_xx = tcal_xx
        self.tcal_yy = tcal_yy
        self.c2K_xx = None
        self.c2K_yy = None
        self.bandpass_xx = None
        self.bandpass_yy = None
        self.theta_off = None
        self.theta_slp = None
        self.theta_obs = None
        self.theta_fit = None


    def calibrate(self):
        """
        """
        self.fill_data()
        # Find calibration factors for auto-correlation products.
        # XX.
        self.dc0xx = DataCalInitAuto(self.freq, 
                                     self.data_xx_on, 
                                     self.data_xx_off,
                                     self.tcal_xx
                                    )
        self.dc0xx.cal_init()
        self.c2K_xx = self.dc0xx.c2K
        # YY.
        self.dc0yy = DataCalInitAuto(self.freq,
                                     self.data_yy_on, 
                                     self.data_yy_off,
                                     self.tcal_yy
                                    )
        self.dc0yy.cal_init()
        self.c2K_yy = self.dc0yy.c2K

        # Now for the cross-correlation products.
        self.dc0cross = DataCalInitCross(self.freq,
                                         self.data_xy_on, self.data_xy_off,
                                         self.data_yx_on, self.data_yx_off,
                                        )
        self.dc0cross.phase_fit()
        self.theta_off = self.dc0cross.theta_off
        self.theta_slp = self.dc0cross.theta_slp
        self.theta_obs = self.dc0cross.theta_obs
        self.theta_fit = self.dc0cross.theta_fit



    def fill_data(self):
        """
        """

        def data_mask(pol, cal):
            """
            pol : {"XX", "YY", "XY", "YX"}
                Polarization product.
            cal : {"T", "F"}
                Noise diode ON ("T") or OFF ("F").
            """
            return (self.table["PLNUM"] == config.pol2indx[pol]) & \
                   (self.table["CAL"] == cal)

        # XX noise diode ON.
        mask = data_mask("XX", "T") 
        self.data_xx_on = self.table["DATA"][mask]
        # XX noise diode OFF.
        mask = data_mask("XX", "F")
        self.data_xx_off = self.table["DATA"][mask]
        # YY, XY and YX.
        self.data_yy_on = self.table["DATA"][data_mask("YY", "T")]
        self.data_yy_off = self.table["DATA"][data_mask("YY", "F")]
        self.data_xy_on = self.table["DATA"][data_mask("XY", "T")]
        self.data_xy_off = self.table["DATA"][data_mask("XY", "F")]
        self.data_yx_on = self.table["DATA"][data_mask("YX", "T")]
        self.data_yx_off = self.table["DATA"][data_mask("YX", "F")]

        # Also fill in the noise diode temperatures if necessary.
        if self.tcal_xx is None:
            self.tcal_xx = self.table["TCAL"][data_mask("XX", "T")].mean()
        if self.tcal_yy is None:
            self.tcal_yy = self.table["TCAL"][data_mask("YY", "T")].mean()



class C2K():
    """
    """

    def __init__(self, c2K, chi=0, chf=-1):
        """
        c2K : array
            Array with c2K values. Should be of shape (ntimes,nchan).
        """

        self.c2K = c2K
        self.chi = chi
        self.chf = chf
        self.outlier_mask = False
        self.c2K_avg = None
        

    def filter_outliers(self, threshold=1):
        """
        """

        diff = np.diff(self.c2K, axis=0)
        stds = diff[:,self.chi:self.chf].std(axis=1)

        bads = np.where(stds > np.median(stds)+threshold*np.std(stds))[0]
        bad_mask = np.zeros(len(self.c2K), dtype=bool)
        bad_mask[bads[1::2]] = True

        self.outlier_mask = bad_mask
        

    def average(self):
        """
        """

        self.c2K_avg = self.c2K[~self.outlier_mask].mean(axis=0)
