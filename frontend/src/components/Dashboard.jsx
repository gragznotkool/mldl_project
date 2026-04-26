import React, { useState, useEffect } from 'react';
import { fetchPrediction, fetchSampleData } from './ApiIntegration';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import { Settings, Droplets, CloudRain, Sun, Cloud, Snowflake, Wind, Activity, Maximize2, AlertTriangle, AlertCircle, HardHat, Calendar, Flag } from 'lucide-react';

const Dashboard = () => {
  // State
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState({
    congestion_index: 0,
    travel_time: 0,
    avg_speed: 0,
    delay: 0,
    congestion_level: "Low",
    hourly_forecast: [],
    factor_contribution: {},
    weekly_pattern: []
  });
  
  // Params
  const [hour, setHour] = useState(8);
  const [day, setDay] = useState(1); // 0=Mon, ...
  const [season, setSeason] = useState('Spring');
  const [weather, setWeather] = useState('Clear');
  const [roadType, setRoadType] = useState('Highway');
  const [specialFactors, setSpecialFactors] = useState([]);
  const [pastValues, setPastValues] = useState([]);

  // Fetch samples on load
  useEffect(() => {
    const init = async () => {
      const sample = await fetchSampleData(1, day);
      if (sample.values) {
        setPastValues(sample.values);
      }
    };
    init();
  }, [day]);

  // Fetch prediction automatically when params change
  useEffect(() => {
    if (pastValues.length === 0) return;

    const generate = async () => {
      setLoading(true);
      const payload = {
        hour, day,
        season, weather,
        road_type: roadType,
        special_factors: specialFactors,
        past_values: pastValues
      };

      const res = await fetchPrediction(payload);
      if (res.success) {
        setData({
          congestion_index: res.metrics.congestion_index,
          travel_time: res.metrics.travel_time,
          avg_speed: res.metrics.avg_speed,
          delay: res.metrics.delay,
          congestion_level: res.congestion_level,
          hourly_forecast: res.charts.hourly_forecast.map((val, i) => ({ time: `${i}:00`, val })),
          factor_contribution: res.charts.factor_contribution,
          weekly_pattern: res.charts.weekly_pattern.map((val, i) => ({ day: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][i], val }))
        });
      }
      setLoading(false);
    };

    // basic debounce
    const timeout = setTimeout(generate, 300);
    return () => clearTimeout(timeout);
  }, [hour, day, season, weather, roadType, specialFactors, pastValues]);

  const toggleFactor = (f) => {
    if (specialFactors.includes(f)) {
      setSpecialFactors(specialFactors.filter(x => x !== f));
    } else {
      setSpecialFactors([...specialFactors, f]);
    }
  };

  const getStatusClass = (level) => {
    if (!level) return '';
    const l = level.toLowerCase();
    if (l.includes('high')) return 'badge-danger';
    if (l.includes('low')) return 'badge-success';
    return 'badge-warning';
  };

  const colors = {
    'Time of day': '#3b82f6',
    'Weather': '#f59e0b',
    'Season': '#10b981',
    'Road type': '#8b5cf6',
    'Special factors': '#ef4444'
  };

  return (
    <div className="flex flex-col gap-6 animate-fade-in pb-10">
      
      {/* TOP CARDS */}
      <div className="grid grid-cols-4 gap-4">
        <div className="card">
          <div className="metric-label">Congestion index</div>
          <div className="metric-val">
            {data.congestion_index}
            <span className={`badge ${getStatusClass(data.congestion_level)}`}>{data.congestion_level}</span>
          </div>
          <div className="metric-sub">out of 100</div>
        </div>
        
        <div className="card">
          <div className="metric-label">Travel time</div>
          <div className="metric-val">{data.travel_time} min</div>
          <div className="metric-sub">vs free-flow (15 min)</div>
        </div>

        <div className="card">
          <div className="metric-label">Avg speed</div>
          <div className="metric-val">{data.avg_speed}</div>
          <div className="metric-sub">km/h</div>
        </div>

        <div className="card relative">
          <div className="absolute top-4 right-4 text-gray-500"><Settings size={18} /></div>
          <div className="metric-label">Delay</div>
          <div className="metric-val">+{data.delay}</div>
          <div className="metric-sub">minutes added</div>
        </div>
      </div>

      {/* MID PANEL */}
      <div className="grid grid-cols-2 gap-4">
        
        {/* Left column */}
        <div className="card flex flex-col gap-6">
          <div>
            <div className="metric-label">TIME & DAY</div>
            <div className="flex justify-between items-center mt-2 mb-1">
              <span style={{fontWeight: 600}}>Hour of day</span>
              <span className="slider-val">{hour}:00</span>
            </div>
            <input 
              type="range" min="0" max="23" 
              className="slider" 
              value={hour} 
              onChange={(e) => setHour(parseInt(e.target.value))} 
            />
          </div>

          <div>
            <div className="metric-label mb-2">DAY OF WEEK</div>
            <div className="btn-select-grid">
              {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((d, i) => (
                <button key={d} className={`btn-select ${day === i ? 'active' : ''}`} onClick={() => setDay(i)}>
                  {d}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="metric-label mb-2">SEASON</div>
            <div className="btn-select-grid">
              {['Spring', 'Summer', 'Autumn', 'Winter'].map((s) => (
                <button key={s} className={`btn-select ${season === s ? 'active' : ''}`} onClick={() => setSeason(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Divider icon wrapper purely for UI aesthetics */}
        {/* We can skip the actual DOM arrow if we want it clean 2 cols */}

        {/* Right column */}
        <div className="card flex flex-col gap-6">
          <div className="metric-label" style={{marginBottom: '-0.5rem'}}>CONDITIONS & ROAD</div>
          
          <div>
            <div className="metric-label text-xs mb-2">WEATHER</div>
            <div className="btn-select-grid">
              {['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow'].map((w) => (
                <button key={w} className={`btn-select ${weather === w ? 'active' : ''}`} onClick={() => setWeather(w)}>
                  {w}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="metric-label text-xs mb-2">ROAD TYPE</div>
            <div className="btn-select-grid">
              {['Highway', 'Arterial', 'Urban', 'Residential'].map((r) => (
                <button key={r} className={`btn-select ${roadType === r ? 'active' : ''}`} onClick={() => setRoadType(r)}>
                  {r}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="metric-label text-xs mb-2">SPECIAL FACTORS</div>
            <div className="btn-select-grid">
              {['Accident', 'Major event', 'Construction', 'School zone', 'Public holiday'].map((f) => (
                <button key={f} className={`btn-select ${specialFactors.includes(f) ? 'active' : ''}`} onClick={() => toggleFactor(f)}>
                  {f}
                </button>
              ))}
            </div>
          </div>
        </div>

      </div>

      {/* CHARTS ROW */}
      <div className="grid grid-cols-2 gap-4">
        
        {/* Forecast Component */}
        <div className="card">
          <div className="chart-title">HOURLY CONGESTION FORECAST</div>
          <div style={{width: '100%', height: '220px'}}>
            <ResponsiveContainer>
              <AreaChart data={data.hourly_forecast} margin={{top: 0, right: 0, left: -20, bottom: 0}}>
                <defs>
                  <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} />
                <Tooltip cursor={{stroke: 'rgba(255,255,255,0.1)', strokeWidth: 2}} contentStyle={{backgroundColor: '#1e1e1e', borderColor: '#333'}} />
                <Area type="monotone" dataKey="val" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill="url(#colorVal)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Factor Breakdown Component */}
        <div className="card">
          <div className="chart-title">FACTOR CONTRIBUTION</div>
          <div className="flex flex-col justify-center h-full pb-4">
            {Object.keys(data.factor_contribution).length > 0 ? (
              Object.entries(data.factor_contribution).map(([key, val]) => {
                let displayKey = key;
                if (key === 'Weather') displayKey = `Weather (${weather.toLowerCase()})`;
                if (key === 'Season') displayKey = `Season (${season.toLowerCase()})`;
                if (key === 'Road type') displayKey = `Road type (${roadType.toLowerCase()})`;
                
                return (
                  <div key={key} className="factor-row">
                    <div className="factor-header">
                      <span>{displayKey}</span>
                      <span>{val}%</span>
                    </div>
                    <div className="factor-bar-bg">
                      <div className="factor-bar-fg" style={{width: `${val}%`, backgroundColor: colors[key] || '#3b82f6'}}></div>
                    </div>
                  </div>
                )
              })
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">Loading factors...</div>
            )}
          </div>
        </div>

      </div>

      {/* WEEKLY PATTERN Component */}
      <div className="card pb-8">
        <div className="chart-title">WEEKLY PATTERN COMPARISON</div>
        <div style={{width: '100%', height: '200px'}}>
          <ResponsiveContainer>
             <BarChart data={data.weekly_pattern} margin={{top: 0, right: 0, left: -20, bottom: 0}} barSize={50}>
               <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
               <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} />
               <YAxis axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} />
               <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{backgroundColor: '#1e1e1e', borderColor: '#333'}} />
               <Bar dataKey="val" radius={[4, 4, 0, 0]}>
                 {
                   data.weekly_pattern.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={index >= 5 ? '#b45309' : '#326c9f'} />
                   ))
                 }
               </Bar>
             </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
    </div>
  );
};

export default Dashboard;
