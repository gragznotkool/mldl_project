const API_URL = '/api';

export const fetchPrediction = async (params) => {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching prediction:', error);
    return { success: false, error: error.message };
  }
};

export const fetchSampleData = async (days = 1, day = 1) => {
  try {
    const response = await fetch(`${API_URL}/sample-data?days=${days}&day=${day}`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching sample data:', error);
    return { success: false, error: error.message };
  }
};

export const fetchVisualization = async () => {
  try {
    const response = await fetch(`${API_URL}/visualize`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching visualization:', error);
    return { success: false, error: error.message };
  }
};
