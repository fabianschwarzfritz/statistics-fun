# Dependencies

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

#%matplotlib inline

with open('.client.id') as f:
  client_id = f.readline()

with open('.client.secret') as f:
  client_secret = f.readline()

def get_spotify_client():
  client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
  spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
  return spotify


def track_ids_from_artist(artist_uri):
  spotify = get_spotify_client()
  
  results = spotify.artist_top_tracks(artist_uri)
  tracks = []
  for track in results['tracks'][:10]:
    tracks.append(track['id'])
  return tracks

def audio_features_by(track_ids):
  spotify = get_spotify_client()

  audio_features = spotify.audio_features(tracks=track_ids)
  return pd.DataFrame(data=audio_features)
  

tracks = track_ids_from_artist('spotify:artist:36QJpDe2go2KgaRleHCDTp')
df = audio_features_by(tracks)
print(df)
