"""Candidate force-based balance readout; native calibration is required."""
from collections import deque
import numpy as np
from native_coupling import array
from balance_display import Screen


class BalanceFilter:
    """Instrument filtering of measured contact force, independent of recipe."""
    def __init__(self):
        self.window=deque();self.time=0.;self.tare_g=None;self.last={}
        self.filtered=deque();self.last_filtered_time=-1.

    def advance_force(self,gross_g,dt):
        self.time+=dt
        if not np.isfinite(gross_g) or not 0<dt<=.02:raise ValueError('Invalid balance measurement')
        self.window.append((self.time,gross_g))
        while self.window and self.window[0][0]<self.time-.75:self.window.popleft()
        values=np.array([v for _,v in self.window])
        mean=float(values.mean());span=float(np.ptp(values))
        if self.time-self.last_filtered_time>=.05-1e-10:
            self.filtered.append((self.time,mean));self.last_filtered_time=self.time
        while self.filtered and self.filtered[0][0]<self.time-.5:self.filtered.popleft()
        filtered_span=float(np.ptp([x for _,x in self.filtered]))
        # Stability follows the filtered instrument reading, not individual
        # solver contact impulses. A changing load still cannot settle early.
        settled=self.time>=1.25 and len(self.filtered)>=8 and mean>50 and filtered_span<.02
        # The benchmark starts with an empty beaker on the pan, pre-tared.
        if self.tare_g is None and settled:self.tare_g=mean
        net=None if self.tare_g is None else mean-self.tare_g
        self.last={'raw_gross_g':gross_g,'mean_gross_g':mean,'tare_g':self.tare_g,
                   'net_g':net,'window_span_g':span,'stable':settled,
                   'filtered_span_g':filtered_span,'filter_window_s':.75,
                   'source':'beaker-to-balance native normal contact force / gravity',
                   'display_resolution_g':.001}


class NativeBalance(BalanceFilter):
    def __init__(self,base):
        super().__init__()
        import omni.usd
        self.base=base;self.sensor=base.scene['balance_contacts']
        self.screen=Screen(omni.usd.get_context().get_stage(),'/World/envs/env_0/digital_balance')

    def advance(self,dt):
        forces=array(self.sensor.data.normal_force_matrix_w)
        if forces.shape!=(1,1,1,3):raise RuntimeError('Unexpected balance contact filter shape')
        self.advance_force(float(forces[0,0,0,2]/9.81*1000),dt)

    def render(self):
        value=self.last.get('net_g')
        self.screen.set_value(None if value is None else round(value,3))
