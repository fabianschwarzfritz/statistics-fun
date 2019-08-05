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

def dataframe_with_artists(artist_ids):
  df = pd.DataFrame()
  counter = 0
  for id in artist_ids:
    track_ids = track_ids_from_artist('spotify:artist:' + id)
    df_artist = audio_features_by(track_ids)
    df_artist.insert(counter, "Artist", counter)
    df = df.append(df_artist)
    counter += 1
  df = df.reset_index(drop=True)
  return df

def plot_feature(df, artist_indexes, properties):
  print('**********************************************')
  print('***** Plot the data **************************')
  print('**********************************************')
  print(df)

  fig, ax = plt.subplots(len(properties), 1)

  property_index = 0
  for property in properties:
    for artist_index in artist_indexes:
      ax[property_index].hist([df.loc[df['Artist'] == artist_index][property]], alpha=0.5, label=artist_index)
      ax[property_index].set_title(property)
      ax[property_index].legend()
    property_index += 1
  plt.tight_layout()
  plt.show()

print('**********************************************')
print('***** All fetched tracks *********************')
print('**********************************************')
lz = '36QJpDe2go2KgaRleHCDTp' # Led Zeppelin
kt = '1oXiuCd5F0DcnmXH5KaM6N' # Kollektiv Turmstrasse
mt = '3nDNDLcZuSto4k9u4AbcLB' # Marteria
artist_ids = [lz, kt, mt]
df = dataframe_with_artists(artist_ids)
print(df)

print('**********************************************')
print('***** Properties used to analyze *************')
print('**********************************************')
# All properties that I might be able to use later.
# df = df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo']]
df = df[['Artist', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo']]
# scaler = MinMaxScaler()
# df[df.columns] = scaler.fit_transform(df[df.columns])

plot_feature(df, [0, 1, 2], ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo'])
