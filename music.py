# Dependencies

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
  

print('**********************************************')
print('***** All fetched tracks *********************')
print('**********************************************')
tracks_led_zeppelin = track_ids_from_artist('spotify:artist:36QJpDe2go2KgaRleHCDTp')
df_led_zeppelin = audio_features_by(tracks_led_zeppelin)
df_led_zeppelin.insert(0, "Artist", 0)

tracks_kollektiv = track_ids_from_artist('spotify:artist:1oXiuCd5F0DcnmXH5KaM6N')
df_kollektiv = audio_features_by(tracks_kollektiv)
df_kollektiv.insert(0, "Artist", 1)

df = pd.concat([df_led_zeppelin, df_kollektiv])
print(df)

print('**********************************************')
print('***** Properties used to analyze *************')
print('**********************************************')
df = df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo']]
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

print('**********************************************')
print('***** Plot the data **************************')
print('**********************************************')
print(df)
df.T.plot()
plt.show()
