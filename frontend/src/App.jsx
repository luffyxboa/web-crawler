import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [query, setQuery] = useState('')
  const [country, setCountry] = useState('')
  const [results, setResults] = useState([])
  const [logs, setLogs] = useState([])
  const [isSearching, setIsSearching] = useState(false)
  const [isEnriching, setIsEnriching] = useState(false)
  const logsEndRef = useRef(null)

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [logs])

  const [limit, setLimit] = useState(10)

  const handleSearch = async () => {
    if (!query) {
      alert('Please enter a search query');
      return;
    }
    if (!country) {
      alert('Please enter a country');
      return;
    }
    setIsSearching(true)
    setResults([])
    setLogs([]) // Clear previous logs

    const response = await fetch(`${API_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, limit, country }),
    });

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const lines = value.split('\n\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'status' || data.type === 'error') {
              setLogs(prev => [...prev, data.message]);
            } else if (data.type === 'company') {
              setResults(prev => [...prev, data.data]);
            } else if (data.type === 'done') {
              setIsSearching(false);
            }
          } catch (e) {
            console.error("Error parsing SSE data", e);
          }
        }
      }
    }
  };

  const handleEnrich = async () => {
    if (results.length === 0) return;
    setIsEnriching(true);
    setLogs(prev => [...prev, 'Starting enrichment pipeline...']);

    const response = await fetch(`${API_URL}/enrich`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ companies: results, country }),
    });

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    const enrichedResults = [];

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const lines = value.split('\\n\\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'status') {
              setLogs(prev => [...prev, data.message]);
            } else if (data.type === 'company') {
              enrichedResults.push(data.data);
            } else if (data.type === 'done') {
              setResults(enrichedResults);
              setIsEnriching(false);
              setLogs(prev => [...prev, data.message]);
            }
          } catch (e) {
            console.error("Error parsing enrichment SSE data", e);
          }
        }
      }
    }
  };

  return (
    <div className="container">
      <h1>AI Web Crawler</h1>

      <div className="search-box">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter search query (e.g. AI startups in SF)"
          disabled={isSearching}
        />
        <input
          type="text"
          value={country}
          onChange={(e) => setCountry(e.target.value)}
          placeholder="Country"
          style={{ width: '150px', marginLeft: '10px' }}
          disabled={isSearching}
          required
        />
        <input
          type="number"
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          placeholder="Limit"
          min="1"
          max="50"
          style={{ width: '80px', marginLeft: '10px' }}
          disabled={isSearching}
        />
        <button onClick={handleSearch} disabled={isSearching}>
          {isSearching ? 'Crawling...' : 'Search'}
        </button>
      </div>

      <div className="status-log">
        <h3>Live Activity Log</h3>
        <div className="logs-container">
          {logs.map((log, index) => (
            <div key={index} className="log-entry">{log}</div>
          ))}
          <div ref={logsEndRef} />
        </div>
      </div>

      {results.length > 0 && (
        <div className="results-section">
          <div className="results-header">
            <h3>Found {results.length} Companies</h3>
            <div className="export-buttons">
              <button onClick={handleEnrich} disabled={isEnriching || isSearching} className="btn enrich-btn">
                {isEnriching ? 'Enriching...' : 'Enrich Results'}
              </button>
              <a href={`${API_URL}/download/csv`} target="_blank" className="btn download-btn">Download CSV</a>
              <a href={`${API_URL}/download/json`} target="_blank" className="btn download-btn">Download JSON</a>
            </div>
          </div>

          <table className="results-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Website</th>
                <th>Email</th>
                <th>Phone</th>
                <th>Address</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {results.map((company, index) => (
                <tr key={index}>
                  <td>{company.name}</td>
                  <td><a href={company.website} target="_blank" rel="noreferrer">{company.website}</a></td>
                  <td>{company.email}</td>
                  <td>{company.phone}</td>
                  <td>{company.address}</td>
                  <td>{company.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default App
