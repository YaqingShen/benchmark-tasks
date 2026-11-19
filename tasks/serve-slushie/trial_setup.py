"""Preparation and spill containment for the simulated slush station."""

PAN_X = (.18, .46)
PAN_Y = (-.12, .29)
TRAY_X = (.315, .440)
TRAY_Y = (-.085, .085)
CONNECTOR_X = (.47, .51)
CONNECTOR_Y = (-.18, -.14)


def prepare_batch(sugar_g=210., water_g=1290.):
    """Prescribed prechill followed by lumped cooling with mixing every ten minutes."""
    mass = sugar_g + water_g
    if sugar_g <= 0 or water_g <= 0:
        raise ValueError('Both ingredients are required')
    temperature = 3.  # Mix is prechilled before entering the freezer.
    history = [temperature]
    # Approximate net cooling power after the initial freeze point. Estimate the
    # ice mass by conserving sugar in the unfrozen solution (ideal cryoscopy).
    def ice_g(temp):
        if temp >= 0:
            return 0.
        return max(0., water_g - (sugar_g / 342.30) * 1.86 / abs(temp) * 1000.)
    def heat_removed(temp):
        return mass * 3.85 * (3. - temp) + ice_g(temp) * 334.
    for mixing_cycle in range(1, 31):
        removed = 75. * mixing_cycle * 600.
        lo, hi = -18., 3.
        for _ in range(40):
            mid = (lo + hi) / 2.
            if heat_removed(mid) > removed:
                lo = mid
            else:
                hi = mid
        temperature = (lo + hi) / 2.
        history.append(round(temperature, 3))
        if -1.7 <= temperature <= -1.2:
            break
        if temperature < -1.7:
            raise ValueError('Batch overshot the serving range')
    else:
        raise ValueError('Batch did not reach the serving range')
    return dict(sugar_g=sugar_g, water_g=water_g, sugar_mass_fraction=sugar_g/mass,
                serving_c=temperature, estimated_ice_g=ice_g(temperature),
                sample_temperatures_c=history, prep_time_min=10*mixing_cycle)


def add_secondary_pan(stage):
    from instrument_visual import box
    box(stage, 'EquipmentConnector', (.49,-.16,.758), (.04,.04,.020), (.28,.24,.19))
    x0, x1 = PAN_X
    y0, y1 = PAN_Y
    z = .748
    color = (.18,.32,.38)
    box(stage,'SecondaryPan',((x0+x1)/2,(y0+y1)/2,z),(x1-x0,y1-y0,.001),color)
    for i,x in enumerate(PAN_X):
        box(stage,'PanEnd'+str(i),(x,(y0+y1)/2,z+.004),(.002,y1-y0,.008),color)
    for i,y in enumerate(PAN_Y):
        box(stage,'PanSide'+str(i),((x0+x1)/2,y,z+.004),(x1-x0,.002,.008),color)


def install_sleeve_contact(cfg):
    """Bind the contact properties of a conformal, zero-thickness cover before simulation starts."""
    from isaaclab.sim.utils import clone
    from pxr import Usd, UsdPhysics, UsdShade
    from yam_arena.robot import asset_link_paths
    original = cfg.spawn.func.__wrapped__

    @clone
    def spawn(prim_path, spawn_cfg, *args, **kwargs):
        arm = original(prim_path, spawn_cfg, *args, **kwargs)
        stage = arm.GetStage()
        mat = UsdShade.Material.Define(stage, prim_path+'/DryPadSleeve')
        friction = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        friction.CreateStaticFrictionAttr(.95)
        friction.CreateDynamicFrictionAttr(.80)
        friction.CreateRestitutionAttr(0.)
        shader = UsdShade.Shader.Define(stage, prim_path+'/DryPadSleeve/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        from pxr import Sdf
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set((.10,.40,.41))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        for tip in ('tip_left','tip_right'):
            root = stage.GetPrimAtPath(prim_path + asset_link_paths()[tip])
            UsdShade.MaterialBindingAPI.Apply(root).Bind(mat, bindingStrength='strongerThanDescendants')
            count = 0
            for child in Usd.PrimRange(root):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(child).Bind(mat, materialPurpose='physics',
                                                                 bindingStrength='strongerThanDescendants')
                    count += 1
            if not count:
                raise RuntimeError('Missing collision surface on '+str(root.GetPath()))
        return arm
    cfg.spawn.func = spawn
