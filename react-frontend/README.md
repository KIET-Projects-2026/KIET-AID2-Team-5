# Traffic Monitoring System - React Frontend

## 🚀 Complete React Migration

This is a **complete React conversion** of the original HTML/CSS/JavaScript Traffic Monitoring System. All functionality has been preserved and enhanced with modern React features.

## ✨ Features Preserved & Enhanced

### Authentication System
- ✅ JWT-based authentication
- ✅ Login and Signup functionality
- ✅ Protected routes
- ✅ Session persistence with localStorage
- ✅ Token verification

### Dashboard Pages
- ✅ **Landing Page** - Hero section with stats and project information
- ✅ **Dashboard** - Real-time traffic overview with live stats
- ✅ **Analytics** - Deep dive analytics with interactive charts
- ✅ **Monitoring** - Live stream management with video feeds

### Advanced Features
- ✅ Real-time WebSocket connections for live updates
- ✅ Multi-stream video monitoring (4 concurrent streams)
- ✅ Stream swapping functionality (click small streams to move to main view)
- ✅ File upload support for video processing
- ✅ YouTube URL support with yt-dlp integration
- ✅ Interactive charts using Chart.js and react-chartjs-2
- ✅ Pagination for violation records
- ✅ Date range filtering for analytics
- ✅ Mobile responsive design
- ✅ Hamburger menu for mobile navigation
- ✅ Toast notifications

### Chart Visualizations
- 📊 Violations Over Time (Line Chart)
- 📊 Violation Type Breakdown (Bar Chart)
- 📊 Violations by Stream (Pie Chart)
- 📊 Speed Distribution (Bar Chart)

## 🏗️ Project Structure

```
react-frontend/
├── public/
│   ├── index.html
│   ├── Logo.png
│   └── image.png
├── src/
│   ├── components/
│   │   ├── AuthModal.js          # Login/Signup modal
│   │   ├── Footer.js              # Footer component
│   │   ├── Navbar.js              # Navigation bar
│   │   └── ProtectedRoute.js     # Route protection HOC
│   ├── context/
│   │   ├── AuthContext.js        # Authentication state management
│   │   └── ConfigContext.js      # Configuration management
│   ├── pages/
│   │   ├── Landing.js            # Landing page with hero
│   │   ├── Dashboard.js          # Live operations center
│   │   ├── Analytics.js          # Analytics with charts
│   │   └── Monitoring.js         # Stream management
│   ├── styles/
│   │   └── styles.css            # All original styles
│   ├── utils/
│   │   └── helpers.js            # Utility functions
│   ├── App.js                     # Main app with routing
│   ├── index.js                   # React entry point
│   └── index.css                  # Global styles
├── package.json
└── README.md
```

## 📦 Installation

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn
- Running backend API (FastAPI server on port 8000)

### Step 1: Navigate to the React Frontend Directory
```bash
cd react-frontend
```

### Step 2: Install Dependencies
```bash
npm install
```

Or with yarn:
```bash
yarn install
```

### Step 3: Start the Development Server
```bash
npm start
```

Or with yarn:
```bash
yarn start
```

The application will open at `http://localhost:3000`

## 🔧 Configuration

The app automatically detects the environment:
- **Local Development**: Uses `http://localhost:8000` for API
- **Production**: Uses `https://traffic-monitoring-api.onrender.com` for API

Configuration is managed by `ConfigContext` and automatically fetched from the backend `/api/config` endpoint.

## 🎯 Usage Guide

### 1. Landing Page
- Visit `http://localhost:3000`
- Click "Get Started - Sign Up" to create an account
- Or click "Already have an account? Login" to sign in

### 2. Authentication
- Complete the signup/login form
- After authentication, you'll be redirected to the Monitoring page
- Your session persists across page refreshes

### 3. Monitoring Page
- **Main View**: Shows the large stream (default: Stream 3)
- **Small Views**: Shows 3 smaller streams (Streams 1, 2, 4)
- **Swap Streams**: Click any small stream to move it to the main view
- **Start Stream**: Enter RTSP/HTTP/YouTube URL and click "Start"
- **Upload Video**: Click the upload area, select a video file, and click "Upload"
- **Stop Stream**: Click "Stop" to terminate any active stream

### 4. Dashboard
- View real-time statistics:
  - Online Streams
  - Today's Traffic
  - Today's Violations
  - Critical Speeding Events
- Live alert feed with recent violations

### 5. Analytics Page
- Select date range (Today, Yesterday, Last Week, etc.)
- View aggregated metrics
- Interactive charts:
  - Violation trends over time
  - Violation type breakdown
  - Violations by camera stream
  - Vehicle speed distribution
- Paginated violation log

## 🚀 Production Build

### Build for Production
```bash
npm run build
```

This creates an optimized production build in the `build/` folder.

### Serve Production Build Locally
```bash
npx serve -s build
```

### Deploy to Production
The build folder can be deployed to any static hosting service:
- Netlify
- Vercel
- GitHub Pages
- AWS S3
- Azure Static Web Apps

## 🔑 Key React Features Used

### Hooks
- `useState` - Component state management
- `useEffect` - Side effects and lifecycle
- `useRef` - DOM references and mutable values
- `useContext` - Global state access
- `useCallback` - Memoized callbacks
- `useNavigate` - Programmatic navigation

### Context API
- `AuthContext` - Authentication state
- `ConfigContext` - Configuration management

### React Router
- Client-side routing
- Protected routes
- Navigation guards

### Third-Party Libraries
- `react-router-dom` - Routing
- `chart.js` & `react-chartjs-2` - Charts
- `@fortawesome/fontawesome-free` - Icons

## 🎨 Styling

All original CSS has been preserved in `src/styles/styles.css`:
- CSS Variables for theming
- Responsive design with media queries
- Mobile-first approach
- Smooth animations and transitions
- Grid and Flexbox layouts

## 📱 Responsive Design

The application is fully responsive:
- **Desktop** (>768px): Full layout with sidebar navigation
- **Tablet** (768px-1024px): Adapted grid layouts
- **Mobile** (<768px): Hamburger menu, stacked layouts, touch-friendly

## 🔐 Security

- JWT token-based authentication
- Tokens stored in localStorage
- Protected routes with automatic redirects
- Token verification on mount
- Secure API communication

## 🐛 Debugging

### Common Issues

**Issue**: "Cannot find module 'react-router-dom'"
**Solution**: Run `npm install`

**Issue**: Blank page on load
**Solution**: Check browser console for errors, ensure backend is running

**Issue**: Streams not loading
**Solution**: Verify backend API is accessible at `http://localhost:8000`

**Issue**: CORS errors
**Solution**: Ensure backend has CORS configured for `http://localhost:3000`

### Development Tools
- React Developer Tools (Chrome/Firefox extension)
- Redux DevTools (if using Redux)
- Network tab for API debugging

## 📊 Performance Optimizations

- Component memoization where needed
- Efficient re-rendering with proper dependencies
- Image optimization for stream frames
- Cleanup of intervals and timeouts
- Lazy loading potential for routes

## 🔄 Migration from Original Code

### What Changed
- ❌ Removed: jQuery, vanilla JS DOM manipulation
- ✅ Added: React components, hooks, context API
- ✅ Improved: Code organization, maintainability
- ✅ Enhanced: Type safety potential (can add TypeScript)

### What Stayed the Same
- ✅ All CSS styles (unchanged)
- ✅ All functionality (preserved)
- ✅ API endpoints (same)
- ✅ User experience (identical)

## 🛠️ Development Commands

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests (if configured)
npm test

# Eject from Create React App (⚠️ irreversible)
npm run eject
```

## 📝 Environment Variables (Optional)

Create `.env` file in root:
```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

Same as original project

## 👥 Credits

- Original HTML/CSS/JS implementation
- React conversion and enhancement
- YOLOv8 for vehicle detection
- FastAPI backend integration

## 🆘 Support

For issues or questions:
- Email: support@tms.com
- Create an issue on GitHub
- Check the documentation

---

**Note**: This is a complete, production-ready React application with all features from the original HTML/CSS/JS version preserved and enhanced. Zero functionality has been lost in the migration.
