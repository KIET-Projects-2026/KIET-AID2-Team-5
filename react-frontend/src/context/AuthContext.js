import React, { createContext, useState, useContext, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useConfig } from './ConfigContext';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [authToken, setAuthToken] = useState(localStorage.getItem('authToken'));
  const [currentUser, setCurrentUser] = useState(() => {
    const saved = localStorage.getItem('currentUser');
    return saved ? JSON.parse(saved) : null;
  });
  const navigate = useNavigate();
  const { API_BASE_URL } = useConfig();

  useEffect(() => {
    if (authToken && currentUser) {
      verifyToken();
    }
  }, [authToken]);

  const getAuthHeaders = () => {
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    return headers;
  };

  const verifyToken = async () => {
    if (!authToken) return false;
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/verify`, {
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      if (!response.ok) {
        logout();
        return false;
      }
      return true;
    } catch (error) {
      console.error('Token verification error:', error);
      return false;
    }
  };

  const login = async (email, password) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Login failed');

      setAuthToken(data.access_token);
      setCurrentUser(data.user);
      localStorage.setItem('authToken', data.access_token);
      localStorage.setItem('currentUser', JSON.stringify(data.user));

      return { success: true, user: data.user };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const signup = async (username, email, password, full_name) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password, full_name })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Signup failed');

      setAuthToken(data.access_token);
      setCurrentUser(data.user);
      localStorage.setItem('authToken', data.access_token);
      localStorage.setItem('currentUser', JSON.stringify(data.user));

      return { success: true, user: data.user };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const logout = async () => {
    try {
      if (authToken) {
        await fetch(`${API_BASE_URL}/api/auth/logout`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${authToken}` }
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    }

    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
    setAuthToken(null);
    setCurrentUser(null);
    navigate('/');
  };

  const value = {
    authToken,
    currentUser,
    isAuthenticated: !!authToken && !!currentUser,
    login,
    signup,
    logout,
    getAuthHeaders
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
