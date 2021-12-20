import numpy as np
import SQ_calcs
import scipy

class Constants(object):
    def __init__(self):
        self.heV = 4.14e-15
        self.c = 2.99792e8
        self.kbeV = 8.6173e-5
        self.keV = 8.6173e-5
        self.h = 6.626e-34
        self.kb = 1.38065e-23
        self.q = 1.60218e-19
        self.k = 1.3806488e-23
        self.T = 300
        self.pixelsize = 5.792e-6
        self.time_btw_frames = 0.1
        self.fps = 10,  # 1 / time_btw_frame
        self.vidfreq = 1
        self.n_same = 13
        self.calibration_factor = 1.7832362718915656e14
        self.bg_cts = 1600
        self.FOV = 136
        self.one_sun_flux = SQ_calcs.one_sun_photon_flux(1.61)
        self.sample_bandgap = 1.61  #eV


    def calc_incident_flux(self, nsun_infos):
        return self.one_sun_flux * nsun_infos

    def sqcalc_vocmax_util(self, nsun_infos):
        """
        what is this?
        """
        return np.asarray([SQ_calcs.VocMax(self.sample_bandgap, ns)
                           for ns in nsun_infos])

const = Constants()


def calibrate_videos(primary_videos, exposure_times, binary_mask=None,
                     keepdims=False, cal_fac = const.calibration_factor):
    """
    Parameters
    ----------
    primary_videos (np.ndarray, dtype=float): primary video input for calibration
        dim = (T_S, T_P, X, Y)

    exposure_times (np.ndarray, dtype=float):
        dim = (T_S, ) or (T_S, T_P)

    binary_mask (np.ndarray, dtype=bool):
        dim = (T_S, X, Y)

    Returns
    -------
    calibrated_primary_videos (np.ndarray): (calibrated) primary video
        dim = (T_S, T_P, X, Y)
        Return a masked array
    """
    if primary_videos.ndim == 3:
        primary_videos = primary_videos[None, ...]

    TS, TP = primary_videos.shape[:2]

    if binary_mask is not None:
        binary_mask = ~(binary_mask.astype(bool))
        if binary_mask.ndim == 2:
            binary_mask = np.repeat(binary_mask[None, ...], TS, axis=0)
        primary_videos = np.ma.array(
            primary_videos, mask=np.repeat(binary_mask[:, None, ...], TP, axis=1))


    if isinstance(exposure_times, np.ndarray):
        exposure_times = exposure_times.reshape(
            exposure_times.shape + \
            tuple(1 for i in range(primary_videos.ndim - exposure_times.ndim)))

    primary_videos = (primary_videos - const.bg_cts) / exposure_times
    primary_videos = primary_videos * cal_fac
    # mask =  np.clip(np.ma.min(primary_videos, axis=1), 0, 1)[:, None, ...]
    # primary_videos = primary_videos * mask
    if keepdims:
        return primary_videos
    else:
        return primary_videos.squeeze()


def calc_plqy_qfls(primary_videos, nsun_infos):
    if not isinstance(nsun_infos, np.ndarray) and not isinstance(nsun_infos, list):
        nsun_infos = np.array([nsun_infos])
    if primary_videos.ndim == 3:
        primary_videos = primary_videos[None, ...]

    incident_flux = const.calc_incident_flux(nsun_infos)
    plqy_videos = primary_videos.mean(1) / incident_flux[:, None, None]
    qfls_videos = const.sqcalc_vocmax_util(nsun_infos)[:, None, None] + \
                  const.keV * const.T * np.ma.log(plqy_videos)

    return plqy_videos.squeeze(), qfls_videos.squeeze()


def generate_2d_window(length, scale=.4):
    xx = signal.get_window(('general_gaussian', 10, length*scale), length)
    two_d_window = np.tile(xx, (length, 1))
    two_d_window = two_d_window * xx[:, None]
    return two_d_window, xx


def generate_2d_round_window(length, scale=.4):
    iii = np.arange(length)
    xxx, yyy = np.meshgrid(iii, iii)
    grid_2d = np.vstack([xxx.flatten(), yyy.flatten()]).T
    wind_2d = np.exp(-0.5*(np.linalg.norm(grid_2d - length//2, axis=1) /\
                           (length * scale)) ** (20)).reshape((length, length))
    return wind_2d


def generate_radii_angle_array(fourier_spectrum, rad_size=200, angle_size=90):
    Ts, num_pixel = fourier_spectrum.shape[:2]
    iii = np.arange(num_pixel)
    radi_interval = np.linspace(0, num_pixel // 2 * np.sqrt(2), rad_size+1)
    radi_start = radi_interval[:-1]
    radi_end = radi_interval[1:]
    angle_interval = np.linspace(-np.pi, np.pi, angle_size + 1)
    angle_start = angle_interval[:-1]
    angle_end = angle_interval[1:]

    radi_angle_array = np.zeros((Ts, rad_size, angle_size))
    size_array = np.zeros((rad_size, angle_size))

    energy_windowed = np.abs(fourier_spectrum) ** 2
    x_coord = iii[None, :] - num_pixel // 2
    y_coord = iii[:, None] - num_pixel // 2
    norm = np.sqrt((x_coord)**2 + (y_coord)**2)
    polar_angle = -np.arctan2(y_coord / (norm+1e-10), x_coord / (norm+1e-10))

    for ix, (rs, re) in enumerate(zip(radi_start, radi_end)):
        mask_start = norm > rs
        mask_end = norm < re
        for ia, (ast, aed) in enumerate(zip(angle_start, angle_end)):
            mask_angle_st = polar_angle > ast
            mask_angle_ed = polar_angle < aed
            ixx_ = np.logical_and.reduce((mask_start, mask_end, mask_angle_st, mask_angle_ed))
            radi_angle_array[:, ix, ia] = energy_windowed[:, ixx_].sum(-1)
            size_array[ix, ia] = ixx_.sum()
    return (radi_angle_array, size_array,
            0.5 * (radi_start + radi_end),
            0.5 * (angle_start + angle_end))


def _find_plqy_75(boolean_time_series, vmin=None):
    time_possible = np.sort(np.where(np.diff(boolean_time_series.astype(int)) == 1)[0])
    if time_possible.shape[0] == 0 or (vmin is not None and np.sum(time_possible > vmin) == 0):
        return boolean_time_series.shape[0]
    elif vmin is not None:
        return time_possible[time_possible > vmin][0]
    else:
        return time_possible[0]


def find_plqy_75(plqy_vids, thres, after_peak=True):
    # TODO: need to optimize this part!!

    time_stopping = np.zeros_like(plqy_vids[0])
    bool_time_series = plqy_vids < thres
    vmin = plqy_vids.mean((-1, -2)).argmax() if after_peak else None
    for i in range(plqy_vids.shape[1]):
        for j in range(plqy_vids.shape[2]):
            time_stopping[i, j] = _find_plqy_75(bool_time_series[:, i, j], vmin=vmin)

    return time_stopping


def find_cont_plqy_75(plqy_vids, thres, time_stopping=None):
    if time_stopping is None:
        time_stopping = find_plqy_75(plqy_vids, thres)
    tstop = time_stopping.flatten().astype(int)
    plqy_flat_all = plqy_vids.reshape((plqy_vids.shape[0], -1))
    ixx_ = tstop < plqy_vids.shape[0] - 1

    plqy_flat_all_ = plqy_flat_all[:, ixx_]
    tstop_ = tstop[ixx_]

    plqy_flat1 = np.array([pq[t] for pq, t in zip(plqy_flat_all_.T, tstop_)])
    plqy_flat2 = np.array([pq[t+1] for pq, t in zip(plqy_flat_all_.T, tstop_)])
    tstop_cont = tstop.copy().astype(float)
    tstop_cont[ixx_] = tstop_ + (thres - plqy_flat1) / (plqy_flat2 - plqy_flat1)

    return tstop_cont.reshape(time_stopping.shape)
