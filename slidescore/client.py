import json
import sys
import requests
import base64
import string
from PIL import Image
import numpy as np
import io
import pickle
from .slidescore import SlideScoreErrorException, SlideScoreResult, APIClient



# url = "http://localhost:10768"
# apitoken = "yourtoken"
# imageid=21813
# client=APIClient(url, apitoken);
# (base_url, cookie) = client.get_image_server_url(imageid);
# Request OpenSeaDragon tiles: format /<level>/<xTileCoord>_<yTileCoord>.jpeg. 
# Size of the tile from metadata, each level is twice the width and height of previous level, level 1 is 1x1 pixel
# response = requests.get(base_url+"/10/1_1.jpeg", cookies=dict(t=cookie));
# you get a JPEG for valid request, empty response for some errors, 5xx or 4xx for others. 
# If you get 503 please call get_image_server_url again and use the new base_url
#TODO: add study id to the __init__ and to the object
class SlidescoreClient(APIClient):
    print_debug = False
    # slidescore_url = "https://rhpc.nki.nl/slidescore"
    slidescore_url = "https://slidescore.nki.nl"
    _save_cache_path = 'slidescore_client_cache_{}.pkl'

    def __init__(self, api_token, server=None, disable_cert_checking=True, log_requests=False):
        if server is None:
            server = SlidescoreClient.slidescore_url
        
        if (server[-1] == "/"):
            server = server[:-1]
        
        super().__init__(server, api_token, disable_cert_checking=disable_cert_checking)
        
        self.log_requests = log_requests
        self._tiles = {}
        self._uc = {}
        self.log = []

    def get_image_metadata(self, imageid):
        
        response = self.perform_request("GetImageMetadata", {
                 "imageId": imageid}, "GET")
 
        rawresp = response.text
        rjson = response.json()
        if (not rjson['success']):
            raise SlideScoreErrorException(rjson['log'])
        return rjson['metadata']

    def _fetch_image_server_url(self, imageid):
        response = self.perform_request("GetTileServer?imageId="+str(imageid), None,  method="GET")
        rjson = response.json()
        return self.end_point.replace("/Api/","/i/"+str(imageid)+"/"+rjson['urlPart']+"/_files"), rjson['cookiePart']
    
    def get_image_server_url(self, imageid):
        u = self._uc.get(imageid, None)
        if u is None:
            u = self._fetch_image_server_url(imageid)
            self._uc[imageid] = u
        return u
        
    def _fetch_tile(self, image_id, tile_coords, zoom_level):
        #TODO: pass the url and the cookie
        #if zoom_level is None:
        #    zoom_level = Basis.min_zoom_level
        #print(image_id, tile_coords)
        verify = False if self.disable_cert_checking else True

        base_url, cookie = self.get_image_server_url(image_id)
        url = base_url + "/{}/{}_{}.jpeg".format(zoom_level, *tile_coords)
        if self.log_requests:
            self.log.append(url)
        response = requests.get(url, cookies=dict(t=cookie), headers=self.request_headers, verify=verify)
        return np.array(Image.open(io.BytesIO(response.content), mode='r'), dtype=np.uint16)

    def get_tile(self, image_id, tile_coords, zoom_level):
        t = self._tiles.get((image_id, *tile_coords, zoom_level), None)
        if t is None:
            t = self._fetch_tile(image_id, tile_coords, zoom_level)
            self._tiles[(image_id, *tile_coords, zoom_level)] = t
        return t
    
    def upload_string_results(self, studyid, results):
        sres = "\n"+"\n".join(results)
        response = self.perform_request("UploadResults", {
                 "studyid": studyid,
                 "results": sres
                 })
        rjson = response.json()
        if (not rjson['success']):
            raise SlideScoreErrorException(rjson['log'])
        return True
    
    def save_cache(self, study_id=None, path=None):
        if path is None:
            path = SlidescoreClient._save_cache_path.format(study_id)
        with open(path, 'wb') as f:
            pickle.dump(self._tiles, f)

    def load_cache(self, study_id=None, path=None, tiles=None):
        if tiles is None:
            if path is None:
                path = SlidescoreClient._save_cache_path.format(study_id)
            with open(path, 'rb') as f:
                self._tiles = pickle.load(f)
        else:
            self._tiles = tiles
