"""Save diagnostic measurements during the continuous scripted run."""

import hashlib
import json


from pack_world import array


def geometry_digest(world):
    return hashlib.sha256((world.root/'assets/mailer.usda').read_bytes()).hexdigest()


def save_checkpoint(driver, phase, holder):
    world = driver.w
    crease = world.box.actuators['passive_scores']
    state = {
        kind: {name: {field: array(value).tolist() for field, value in fields.items()}
               for name, fields in assets.items()}
        for kind, assets in world.base.scene.get_state().items()
    }
    data = {
        'geometry': geometry_digest(world), 'phase': phase, 'holder': holder,
        'held_panel': driver.lid_hold_panels[holder],
        'step': world.steps, 'scene': state,
        'plastic_rotation': array(crease.plastic_rotation).tolist(),
        'plastic_dissipation': array(crease.plastic_dissipation).tolist(),
        'action': driver.action.tolist(), 'home': driver.home.tolist(),
        'base_position': world.last['box']['base_pose'][:3],
        'lid_angle_deg': world.last['box']['angles_deg']['lid'],
    }
    destination = world.output/'checkpoint.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(destination)
