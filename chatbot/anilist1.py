import requests
import json

# Here we define our query as a multi-line string
query = '''
query ($id: Int) { # Define which variables will be used in the query (id)
  Media (id: $id, type: ANIME) { # Insert our variables into the query arguments (id) (type: ANIME is hard-coded in the query)
    id
    title {
      romaji
      english
      native
    }
  }
}
'''

character_query = '''
{
  Media {
    characters(page: 1, perPage: 10) {
      pageInfo {
        total
        perPage
        currentPage
        lastPage
        hasNextPage
      }
      edges {
        node { # The character data node
          id
          name {
            first
            last
          }
        }
        role
        voiceActors (language: JAPANESE) { # Array of voice actors of this character for the anime
          id
          name {
            first
            last
          }
        }
      }
    }
  }
}
'''

genreCollection_query = '''
{ # Define which variables will be used in the query (id)
  GenreCollection
}
'''


# Define our query variables and values that will be used in the query request
variables = {
    'id': 15125
}

url = 'https://graphql.anilist.co'

# Make the HTTP Api request
response = requests.post(url, json={'query': genreCollection_query})
py_response = json.loads(response.content)
print(py_response['data']['GenreCollection'])

# GenreCollection provides possible genres
