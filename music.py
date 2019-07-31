# Dependencies

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#%matplotlib inline
print("**************************")
print("***** Authentication *****")
print("**************************")

with open('.client.id') as f:
  client_id = f.readline()

with open('.client.secret') as f:
  client_secret = f.readline()

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

lz_uri = 'spotify:artist:36QJpDe2go2KgaRleHCDTp'

spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
results = spotify.artist_top_tracks(lz_uri)

for track in results['tracks'][:10]:
  print('track    : ' + track['name'])
  print('audio    : ' + track['preview_url'])
  print('cover art: ' + track['album']['images'][0]['url'])
  print

print("******************************")
print("***** Get audio features *****")
print("******************************")

tracks = []
for track in results['tracks'][:10]:
  tracks.append(track['id'])

print("******************************")
print("***** Tracks analyzed ********")
print("******************************")
print(tracks)

print("*****************************")
print("***** Audio features ********")
print("*****************************")
print(spotify.audio_features(tracks=tracks))

