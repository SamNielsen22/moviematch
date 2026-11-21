// Movie Match: enter your Letterboxd username, then say what you're in the mood for, and get 20 films.
// The films come from the Python API in scripts/web/server.py.

import { useState } from 'react'

function Film({ film, rank }) {
  const [poster, setPoster] = useState(film.poster_url)
  const [askedForNew, setAskedForNew] = useState(false)

  // Some saved poster links are dead. Ask the server for a new one, once.
  async function replacePoster() {
    if (askedForNew) return setPoster(null)
    setAskedForNew(true)
    const response = await fetch(`/api/poster?slug=${film.slug}`)
    const data = await response.json()
    setPoster(response.ok ? data.poster_url : null)
  }

  return (
    <li className="film">
      {poster ? <img src={poster} alt="" onError={replacePoster} /> : <div className="no-poster" />}
      <div>
        <a href={`https://letterboxd.com/film/${film.slug}/`} target="_blank">
          {rank}. {film.title} ({film.year})
        </a>
        <p className="details">{film.genres}{film.rating && ` ★ ${film.rating}`}</p>
        <p>{film.synopsis}</p>
      </div>
    </li>
  )
}

function App() {
  const [step, setStep] = useState('user') // 'user', then 'mood'
  const [username, setUsername] = useState('')
  const [mood, setMood] = useState('')
  const [films, setFilms] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Call the API. Returns its data, or null after showing what went wrong.
  async function getJson(url) {
    setError('')
    setLoading(true)
    try {
      const response = await fetch(url)
      const data = await response.json()
      setLoading(false)
      if (!response.ok) {
        setError(data.detail) // the error message from the server
        return null
      }
      return data
    } catch {
      setLoading(false)
      setError("Couldn't reach the server")
      return null
    }
  }

  async function submitUser(event) {
    event.preventDefault() // stop the browser from reloading the page
    const profile = await getJson(`/api/profile?username=${encodeURIComponent(username.trim())}`)
    if (profile) setStep('mood')
  }

  function skip() {
    setUsername('')
    setError('')
    setStep('mood')
  }

  function back() {
    setFilms([])
    setError('')
    setStep('user')
  }

  async function submitMood(event) {
    event.preventDefault()
    setFilms([])
    const data = await getJson(`/api/match?${new URLSearchParams({ username: username.trim(), mood })}`)
    if (data) setFilms(data)
  }

  return (
    <main>
      <h1>Movie Match</h1>

      {step === 'user' && (
        <form onSubmit={submitUser}>
          <label htmlFor="username">Letterboxd username</label>
          <input id="username" required value={username} onChange={(e) => setUsername(e.target.value)} />
          <div className="buttons">
            <button disabled={loading}>Next</button>
            <button type="button" className="link" onClick={skip}>Skip</button>
          </div>
        </form>
      )}

      {step === 'mood' && (
        <form onSubmit={submitMood}>
          <p className="user">
            {username || 'No username'} <button type="button" className="link" onClick={back}>change</button>
          </p>
          {/* with a username the mood can be left blank: the films then come from taste alone */}
          <label htmlFor="mood">What are you in the mood for?{username && ' (optional)'}</label>
          <textarea id="mood" rows={3} required={!username} value={mood} onChange={(e) => setMood(e.target.value)} />
          <div className="buttons">
            <button disabled={loading}>Find films</button>
          </div>
        </form>
      )}

      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      <ol className="films">
        {films.map((film, i) => <Film key={film.slug} film={film} rank={i + 1} />)}
      </ol>
    </main>
  )
}

export default App
