import config as config
import json
# import sys
# sys.path.append(config.base_path)


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class SetDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def to_sets(o):
        if isinstance(o, list):
            return {SetDecoder.to_sets(v) for v in o}
        elif isinstance(o, dict):
            return {k: SetDecoder.to_sets(v) for k, v in o.items()}
        return o
    
    def object_hook(self, obj):
        return SetDecoder.to_sets(obj)


class TaskTracker:
    progress_template = {'cd_doing': set(), 'cd_done': set(), 'cc_doing': set(), 'cc_done': set()}
    
    def __init__(self, all_slide_ids, progress_file_path):
        
        self.all_slide_ids = all_slide_ids
        self.progress_file_path = progress_file_path
        # 
        self._num_running_tasks = 0
        #        
        try:
            f = open(progress_file_path, 'r')
            self.progress = json.load(f, cls=SetDecoder)
            self.clear_doing()
            print('found previous progress file: ', progress_file_path, flush=True)
            f.close()
        except FileNotFoundError:
            self.progress = TaskTracker.progress_template
    
    def update_all_ids(self, new_slide_ids):
        self.all_slide_ids = new_slide_ids
    
    def get_next_slide(self):
        cc_due = [i for i in self.progress['cd_done'] if not (i in self.progress['cc_doing'].union(self.progress['cc_done']))]
        cd_due = [i for i in self.all_slide_ids if not (i in self.progress['cd_doing'].union(self.progress['cd_done']))]
        if len(cc_due) > 0:
            return cc_due[0], 'cc'
        elif len(cd_due) > 0:
            return cd_due[0], 'cd'
        else:
            return None, 'finished'

    def start_cd(self, slide_id):
        print('start cd: ', slide_id, flush=True)
        if slide_id in [i for v in self.progress.values() for i in v]:
            raise ValueError('slide already registered: ' + str(slide_id))
        
        self._num_running_tasks += 1
        self.progress['cd_doing'] = self.progress['cd_doing'].union([slide_id])
        self._persist()
        
    def finish_cd(self, slide_id):
        if slide_id not in self.progress['cd_doing']:
            raise ValueError('slide not registered for cd: ' + str(slide_id))
        
        self.progress['cd_doing'] = self.progress['cd_doing'] - set([slide_id])
        self.progress['cd_done'] = self.progress['cd_done'].union([slide_id])
        self._num_running_tasks -= 1
        self._persist()

        print('finish cd: ', slide_id, flush=True)
    
    def start_cc(self, slide_id):
        print('start cc: ', slide_id, flush=True)
        if slide_id in self.progress['cc_doing'].union(self.progress['cc_done']):
            raise ValueError('slide already registered: ' + str(slide_id))
        elif slide_id not in self.progress['cd_done']:
            raise ValueError('cd not registered for slide: ' + str(slide_id))
        
        self._num_running_tasks += 1
        self.progress['cc_doing'] = self.progress['cc_doing'].union([slide_id])
        self._persist()
    
    def finish_cc(self, slide_id):
        if slide_id not in self.progress['cc_doing']:
            raise ValueError('slide not registered for cc: ' + str(slide_id))
            
        self.progress['cc_doing'] = self.progress['cc_doing'] - set([slide_id])
        self.progress['cc_done'] = self.progress['cc_done'].union([slide_id])
        self._num_running_tasks -= 1
        self._persist()            
        print('finish cc: ', slide_id, flush=True)
    
    def clear_doing(self, task=None):
        if task == 'cc':
            self.progress['cc_doing'] = set()
        elif task == 'cd':
            self.progress['cd_doing'] = set()
        elif task is None:
            self.progress['cc_doing'] = set()
            self.progress['cd_doing'] = set()
        else:
            raise ValueError(' Unknown task: ' + str(task))

    def clear_done(self, task=None):
        if task == 'cc':
            self.progress['cc_done'] = set()
        elif task == 'cd':
            self.progress['cd_done'] = set()
        elif task is None:
            self.progress['cc_done'] = set()
            self.progress['cd_done'] = set()
        else:
            raise ValueError(' Unknown task: ' + str(task))
        self._persist()
            
    def _persist(self):
        with open(self.progress_file_path, 'w') as f:
            json.dump(self.progress, f, cls=SetEncoder)
