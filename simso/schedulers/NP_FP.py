"""
Non-preemptive Fixed Priority algorithm for uniprocessor architectures.
"""
from simso.core import Scheduler
from simso.schedulers import scheduler

@scheduler("simso.schedulers.NP_FP")
class NP_FP(Scheduler):
    def init(self):
        self.ready_list = []

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()

    def on_terminated(self, job):
        self.ready_list.remove(job)
        job.cpu.resched()

    def schedule(self, cpu):
        if cpu.was_running is None:
            if self.ready_list:
                # job with the highest priority
                job = min(self.ready_list, key=lambda x: x.period)
            else:
                job = None

            return job, cpu
        else:
            if cpu.was_running.end_date is not None:
                if self.ready_list:
                    # job with the highest priority
                    job = max(self.ready_list, key=lambda x: x.data['priority'])
                else:
                    job = None

                return job, cpu

            else:
                return cpu.was_running, cpu
