"""Task-process compatibility for the installed Warp-first Isaac Lab snapshot.

Some COM kernels pass ProxyArray to Warp without selecting its documented .warp
view. Unwrap only those wrappers at the launch-cache boundary; preserve the same
underlying array, kernel, arguments and device. Never edit the shared package.
"""
def install():
    from isaaclab.utils.warp.launch_cache import _WarpLaunchCache
    from isaaclab.utils.warp.proxy_array import ProxyArray
    original=_WarpLaunchCache.launch
    if getattr(original,'_particle_proxy_compat',False):return
    def launch(self,key,kernel,*,dim,inputs,outputs):
        unwrap=lambda x:x.warp if isinstance(x,ProxyArray) else x
        return original(self,key,kernel,dim=dim,inputs=[unwrap(x) for x in inputs],outputs=[unwrap(x) for x in outputs])
    launch._particle_proxy_compat=True
    _WarpLaunchCache.launch=launch
    print('BATTER_BACKEND_COMPAT ProxyArray .warp zero-copy unpacking enabled in task process',flush=True)

    # The same installed snapshot also passes proxies to uncached kernels.
    # Unwrap at the public launch boundary too; no allocation or data write.
    import warp as wp
    from functools import wraps
    raw_launch=wp.launch
    if not getattr(raw_launch,'_particle_proxy_compat',False):
        def unwrap_arg(x):
            if isinstance(x,ProxyArray):return x.warp
            if isinstance(x,list):return [unwrap_arg(y) for y in x]
            if isinstance(x,tuple):return tuple(unwrap_arg(y) for y in x)
            return x
        @wraps(raw_launch)
        def native_launch(*args,**kwargs):
            return raw_launch(*(unwrap_arg(x) for x in args),**{k:unwrap_arg(v) for k,v in kwargs.items()})
        native_launch._particle_proxy_compat=True
        wp.launch=native_launch
