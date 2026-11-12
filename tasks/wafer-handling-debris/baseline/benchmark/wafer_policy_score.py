"""A single binary task score; partial handling remains visible as diagnostics."""
import copy

def combine_score(handling,particles):
    result=copy.deepcopy(handling)
    result['handling_progress_score']=result['score']
    result['particle_check']=copy.deepcopy(particles)
    checks=particles['checks']
    integrity=('initially_on_wafer','no_loss_before_air','all_particles_present_and_unchanged',
               'force_equations_replayed','wafer_reaction_replayed','bond_states_replayed',
               'no_other_external_wrench','no_particle_pose_writes')
    for key in integrity:
        if not checks.get(key):
            result['invalid_reasons'].append('Particle check failed: '+key)
    passed=bool(handling['success'] and handling['evidence_complete'] and particles['passed'] and not result['invalid_reasons'])
    result.update(score=float(passed),success=passed)
    if not passed:result['completion_time_s']=None
    return result
