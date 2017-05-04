#!/usr/bin/env python
import os
import sys
import yaml
import numpy as np
import logging
from collections import namedtuple, OrderedDict

import inheritance
import scenarios
import sinks


DataMetadata = namedtuple('DataMetadata', 'name desc dims')
ModelContext = namedtuple('ModelContext', 'data parm ctrl meta logger')
ScenarioContext = namedtuple('ScenarioContext', 'parm')


def load_model_tensor(fpath):
    obj = np.load(fpath)
    #sparse matrices are saved as 0-d object array :(
    arr = obj['array']
    if arr.shape == () and arr.dtype == object:
        arr = arr.item()
    msg = "NDArray shape and dims meta data are not aligned '{}'".format(fpath)
    assert arr.shape == tuple(map(len, obj['dims'])), msg
    return arr, DataMetadata(name=str(obj['name']), desc=str(obj['desc']),
                             dims=np.squeeze(obj['dims']))


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s: %(message)s')
logger = logging.getLogger()

#TODO: type conversion should be explicit
def convert_type(ob):
    r = ob
    try:
        r = np.float32(r)
    except ValueError:
        pass #ignore
    except TypeError:
        pass #ignore
    except AttributeError:
        pass #ignore
    return r


def merge_namedtuple(base, overriding):
    #list of unique keys while maintaining order (kludge)
    keys = list(OrderedDict.fromkeys(base._fields+overriding._fields))
    elems = []
    for k in keys:
        if k in overriding._fields:
            elems.append(overriding.__dict__[k])
        else:
            elems.append(base.__dict__[k])

    t = namedtuple(type(base).__name__, keys)
    return t(*elems)


def update_context(ctx, mod_ctx):
    mod_ctx = mod_ctx if type(mod_ctx) is list else [mod_ctx]

    data = ctx.data
    parm = ctx.parm
    ctrl = ctx.ctrl
    meta = ctx.meta

    for sub_ctx in mod_ctx:
        ctx_name = type(sub_ctx).__name__
        if ctx_name == 'data':
            data = merge_namedtuple(data, sub_ctx)
        elif ctx_name == 'parm':
            parm = merge_namedtuple(parm, sub_ctx)
        elif ctx_name == 'ctrl':
            ctrl = merge_namedtuple(ctrl, sub_ctx)
        elif ctx_name == 'meta':
            meta = merge_namedtuple(meta, sub_ctx)
        else:
            raise ValueError("Unrecognised modified sub context '%s' " % ctx_name)

    return ModelContext(data=data, parm=parm , ctrl=ctrl, meta=meta, logger=ctx.logger)


class SimulatorAdapter(object):


    def __init__(self, config):
        self.inputs = {d['dataset']: d['filepath'] for d in config['model_data_files']}
        self.model_parameters = [(p['name'], p['value']) for p in config['model_parameters']]
        self.control_parameters = config['control_parameters']
        if 'seed' not in self.control_parameters:
            self.control_parameters['seed'] = np.random.randint(2**32-1)
        self.output_filename = 'output.h5'


    def run(self):

        #intialisation of context
        tensors = map(load_model_tensor, self.inputs.values())
        arrays, metadata = zip(*tensors) if tensors else ([], [])
        data_names = map(lambda x: str(x.name), metadata)
        ModelData = namedtuple('ModelData', ' '.join(data_names))
        data = ModelData(*arrays)
        ModelMeta = namedtuple('ModelMeta', ' '.join(data_names))
        meta = ModelMeta(*metadata)

        #TODO: would be nice to have type information
        parm_names, parm_values = zip(*self.model_parameters)
        ModelParm = namedtuple('ModelParm', ' '.join(parm_names))
        parm = ModelParm(*map(convert_type, parm_values))

        #TODO: again it would be nice to parse type information
        temp = list(self.control_parameters.iteritems())
        temp += [('random_state', np.random.RandomState(seed=int(self.control_parameters['seed'])))] #add random state
        ctrl_names, ctrl_values = zip(*temp)
        ModelCtrl = namedtuple('ModelCtrl', ' '.join(ctrl_names))
        ctrl = ModelCtrl(*map(convert_type, ctrl_values))

        ctx = ModelContext(data=data, parm=parm , ctrl=ctrl, meta=meta, logger=logger)

        #Apply base data mutation
        mod_ctx = scenarios.base(ctx)
        ctx = update_context(ctx, mod_ctx)

        store = sinks.HDFSink(ctx, self.output_filename)
        sink = sinks.CompositeSink(
            map(lambda s: getattr(sinks, s)(ctx, store), ctx.ctrl.sinks)
        )
        inheritance.simulate(ctx, sink)

        ctx.logger.info('simulation finished')

        #post-processing
        sink.flush()
        store.flush()


if __name__ == '__main__':

    assert len(sys.argv) == 2, 'missing configuration file'
    config = yaml.load(open(sys.argv[1], 'r'))

    sim = SimulatorAdapter(config)
    sim.run()

    sys.exit(0)
