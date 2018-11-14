# ADAPTED FROM https://github.com/openai/gym-http-api
import requests
import six.moves.urllib.parse as urlparse
import json
import os
import pkg_resources
import sys
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

features = sorted(['joint_pos','joint_vel','joint_acc','body_pos','body_vel','body_acc',
            'body_pos_rot','body_vel_rot','body_acc_rot','misc'])
body_parts = sorted(['femur_r','pros_tibia_r','pros_foot_r','femur_l','tibia_l','talus_l','calcn_l','toes_l','torso','head'])
def concatenate_state(state):
    result = []
    parts = sorted(state.keys())
    for part in parts:
      if part[:4] != 'body':
        sub_parts = sorted(state[part].keys())
        for sub_part in sub_parts:
            if type(state[part][sub_part]) == dict:
               typek = sorted(state[part][sub_part].keys())
               for k in typek:
                   result.append(state[part][sub_part][k])
            else:
               result = result + state[part][sub_part].copy()
      else:

        pelvis = state[part]['pelvis']
        result = result + pelvis.copy()
        for sub_part in body_parts:
            result = result + (np.array(state[part][sub_part]) - np.array(pelvis)).tolist()
    return result

def trapezoid(x,a,b,c,d):
    return max(min((x-a)/(b-a),5,(d-x)/(d-c)),-5)
class Client(object):
    """
    Gym client to interface with gym_http_server
    """
    def __init__(self, remote_base):
        self.remote_base = remote_base
        self.session = requests.Session()
        self.session.headers.update({'Content-type': 'application/json'})
        self.instance_id = None

    def _parse_server_error_or_raise_for_status(self, resp):
        j = {}
        try:
            j = resp.json()
        except:
            # Most likely json parse failed because of network error, not server error (server
            # sends its errors in json). Don't let parse exception go up, but rather raise default
            # error.
            resp.raise_for_status()
        if resp.status_code != 200 and "message" in j:  # descriptive message from server side
            raise ServerError(message=j["message"], status_code=resp.status_code)
        resp.raise_for_status()
        return j

    def _post_request(self, route, data):
        url = urlparse.urljoin(self.remote_base, route)
        logger.info("POST {}\n{}".format(url, json.dumps(data)))
        resp = self.session.post(urlparse.urljoin(self.remote_base, route),
                            data=json.dumps(data))
        return self._parse_server_error_or_raise_for_status(resp)

    def _get_request(self, route):
        url = urlparse.urljoin(self.remote_base, route)
        logger.info("GET {}".format(url))
        resp = self.session.get(url)
        return self._parse_server_error_or_raise_for_status(resp)

    def env_create(self, token, env_id = "Run"):
        route = '/v1/envs/'
        data = {'env_id': env_id,
                'token': token,
                'version': "2.0.0" }
        try:
            resp = self._post_request(route, data)
        except ServerError as e:
            sys.exit(e.message)
        self.instance_id = resp['instance_id']
        self.env_monitor_start("tmp", force=True)
        return self.env_reset()

    def env_reset(self):
        route = '/v1/envs/{}/reset/'.format(self.instance_id)
        resp = self._post_request(route, None)
        observation = resp['observation']
        if not observation:
            return None
        observation = self.obg(np.array(concatenate_state(observation)))
        return observation
    def obg(self,plain_obs):
                plain_obs[plain_obs > 1000] = 0
                plain_obs[plain_obs < -1000] = 0
                return plain_obs
    def env_step(self, action, render=False):
        route = '/v1/envs/{}/step/'.format(self.instance_id)
        data = {'action': action, 'render': render}
        resp = self._post_request(route, data)
        observation = resp['observation']
        reward = resp['reward']
        done = resp['done']
        info = resp['info']
        observation = self.obg(np.array(concatenate_state(observation)))
        return [observation, reward, done, info]

    def env_monitor_start(self, directory,
                              force=False, resume=False, video_callable=False):
        route = '/v1/envs/{}/monitor/start/'.format(self.instance_id)
        data = {'directory': directory,
                'force': force,
                'resume': resume,
                'video_callable': video_callable}
        self._post_request(route, data)

    def submit(self):
        route = '/v1/envs/{}/monitor/close/'.format(self.instance_id)
        result = self._post_request(route, None)
        if result['reward']:
            print("Your total reward from this submission: %f" % result['reward'])
        else:
            print("There was an error in your submission. Please contact administrators.")
        route = '/v1/envs/{}/close/'.format(self.instance_id)
        self.env_close()

    def env_close(self):
        route = '/v1/envs/{}/close/'.format(self.instance_id)
        self._post_request(route, None)

class ServerError(Exception):
    def __init__(self, message, status_code=None):
        Exception.__init__(self)
        print('ServerErrorPrint',message,status_code)
        self.message = message
        if status_code is not None:
            self.status_code = status_code

