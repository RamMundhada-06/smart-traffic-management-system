
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const fetchTrafficStatus = async () => {
  const response = await axios.get(`${API_BASE_URL}/traffic`);
  return response.data;
};

export const triggerManualOverride = async (lane) => {
  const response = await axios.post(`${API_BASE_URL}/manual`, {
    lane,
    action: 'enable',
  });
  return response.data;
};

export const disableManualOverride = async () => {
  const response = await axios.post(`${API_BASE_URL}/manual`, {
    action: 'disable',
  });
  return response.data;
};
