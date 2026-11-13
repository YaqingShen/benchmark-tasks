"""Deterministic component exercise. NOT a robot demo or a policy evaluation."""
import argparse
import json
from pathlib import Path
from sim import Formulation


def stage(env, label, pause=.4):
    env.stage = label
    env.log("stage", label=label)
    env.record()
    env.wait(pause)


def bulb(env, tool, compression):
    for _ in range(1000):
        if abs(env.pipettes[tool].compression - compression) < 1e-9:
            return
        env.bulb_step(tool, compression)
    raise RuntimeError("Bulb did not reach command")


def aspirate(env, tool, source, amount):
    env.set_tip(tool, None)
    bulb(env, tool, 1.)
    env.set_tip(tool, source, submerged=True)
    bulb(env, tool, max(0., 1.-amount/5.))


def dispense(env, tool, destination):
    env.set_tip(tool, destination)
    bulb(env, tool, 1.)
    env.set_tip(tool, None)


def coarse_mass(env,bottle,target_total_g):
    while True:
        reading=env.balance()
        if reading is None:raise RuntimeError('Balance is not settled')
        if reading>=target_total_g:break
        env.pour_step(bottle,'beaker',75.,.02)
    env.vessels[bottle].tilt_deg=0.
    env.wait(.5)


def correct_mass(env,tool,bottle,target_total_g,allow_remove):
    for _ in range(30):
        reading=env.balance()
        if reading is None:raise RuntimeError('Balance is not settled')
        difference=target_total_g-reading
        if abs(difference)<=env.cfg['balance_resolution_g']/2:return
        if difference<0 and not allow_remove:
            raise RuntimeError('B overshoot: removing mixture would also remove A')
        amount=min(env.pipettes[tool].capacity_ml,abs(difference)/env.cfg['density_g_ml'][tool[-1]])
        if difference>0:
            aspirate(env,tool,bottle,amount);dispense(env,tool,'beaker')
        else:
            aspirate(env,tool,'beaker',amount);dispense(env,tool,'waste')
        env.wait(.5)
    raise RuntimeError('Mass correction did not converge')


def run(initial_mass=8.6):
    e=Formulation(initial_beaker_g=initial_mass);e.wait(1.)
    target=e.cfg['targets_g']['A'];bottle='bottle_A';tool='pipette_A'
    stage(e,'Read the starting mass; preserve the empty-beaker tare')
    if initial_mass<target:
        stage(e,'Use the pipette to add small amounts from stock')
    else:
        stage(e,'Use the pipette to remove excess liquid to waste')
    correct_mass(e,tool,bottle,target,allow_remove=True)
    e.log('mass_checkpoint',ingredient='A',observed_g=e.balance())
    stage(e,'Discard the used pipette in the bin');e.discard_pipette(tool)
    stage(e,'Complete: 10 g in the beaker and the used pipette in the bin')
    e.wait(2.);e.record();return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("../../outputs/formulation-prototype"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = run()
    result = {"kind": "component-model exercise", "config": env.cfg,
              "duration_s": env.time, "result": env.score(),
              "events": env.events, "trace": env.trace}
    (args.out/"component-run.json").write_text(json.dumps(result, indent=2))
    suite=[]
    for initial in env.cfg['starting_masses_g']:
        trial=run(initial)
        item={"kind":"component-model exercise","config":trial.cfg,"initial_beaker_g":initial,
              "duration_s":trial.time,"result":trial.score(),"events":trial.events,"trace":trial.trace}
        tag='underfilled' if initial<trial.cfg['targets_g']['A'] else 'overfilled'
        (args.out/('component-'+tag+'.json')).write_text(json.dumps(item,indent=2))
        suite.append({"case":tag,"initial_beaker_g":initial,"result":trial.score()})
    (args.out/'component-correction-cases.json').write_text(json.dumps(suite,indent=2))
    print(json.dumps(suite,indent=2))
