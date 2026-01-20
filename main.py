from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
from typing import List
import os
from dotenv import load_dotenv

from database import get_db, init_db, User, Portfolio, UserPreferences, DailyRecommendation, NewsArticle
from auth import (
    get_password_hash, 
    verify_password, 
    create_access_token, 
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from pydantic import BaseModel, EmailStr

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Stock Predictor API", version="1.0.0")

# CORS middleware - MUST be before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()


# ============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ============================================================================

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    risk_profile: str = "medium"
    capital: float = 10000.0


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    risk_profile: str
    capital: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class PortfolioItem(BaseModel):
    ticker: str
    shares: float
    avg_cost: float


class PortfolioResponse(BaseModel):
    id: int
    ticker: str
    shares: float
    avg_cost: float
    added_at: datetime
    
    class Config:
        from_attributes = True


class RecommendationResponse(BaseModel):
    ticker: str
    company_name: str
    current_price: float
    prediction_1d: float
    prediction_5d: float
    confidence: float
    expected_return: float
    risk_score: float
    action: str
    reasoning: dict
    
    class Config:
        from_attributes = True


# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.post("/api/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    new_user = User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        risk_profile=user.risk_profile,
        capital=user.capital
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create default preferences
    preferences = UserPreferences(user_id=new_user.id)
    db.add(preferences)
    db.commit()
    
    return new_user


@app.post("/api/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token"""
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info"""
    return current_user


@app.put("/api/auth/me", response_model=UserResponse)
def update_user(
    risk_profile: str = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user settings"""
    if risk_profile and risk_profile in ["conservative", "medium", "aggressive"]:
        current_user.risk_profile = risk_profile
        db.commit()
        db.refresh(current_user)
    
    return current_user


# ============================================================================
# PORTFOLIO ENDPOINTS
# ============================================================================

@app.get("/api/portfolio", response_model=List[PortfolioResponse])
def get_portfolio(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Get user's portfolio"""
    portfolios = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).all()
    return portfolios


@app.post("/api/portfolio", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
def add_to_portfolio(
    item: PortfolioItem,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add stock to portfolio"""
    # Check if already exists
    existing = db.query(Portfolio).filter(
        Portfolio.user_id == current_user.id,
        Portfolio.ticker == item.ticker.upper()
    ).first()
    
    if existing:
        # Update existing
        existing.shares = item.shares
        existing.avg_cost = item.avg_cost
        existing.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing
    else:
        # Create new
        new_item = Portfolio(
            user_id=current_user.id,
            ticker=item.ticker.upper(),
            shares=item.shares,
            avg_cost=item.avg_cost
        )
        db.add(new_item)
        db.commit()
        db.refresh(new_item)
        return new_item


@app.delete("/api/portfolio/{ticker}")
def remove_from_portfolio(
    ticker: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Remove stock from portfolio"""
    item = db.query(Portfolio).filter(
        Portfolio.user_id == current_user.id,
        Portfolio.ticker == ticker.upper()
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Stock not in portfolio")
    
    db.delete(item)
    db.commit()
    return {"message": f"Removed {ticker} from portfolio"}


@app.get("/api/portfolio/value")
def get_portfolio_value(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Calculate total portfolio value with current prices"""
    import yfinance as yf
    
    portfolio_items = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).all()
    
    if not portfolio_items:
        return {
            "total_value": 0,
            "total_cost": 0,
            "total_gain": 0,
            "total_gain_percent": 0,
            "positions": []
        }
    
    positions = []
    total_value = 0
    total_cost = 0
    
    for item in portfolio_items:
        try:
            # Fetch current price from yfinance
            ticker = yf.Ticker(item.ticker)
            current_price = ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice', 0)
            
            if current_price == 0:
                # Fallback: try to get from history
                hist = ticker.history(period='1d')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            
            current_value = current_price * item.shares
            cost_basis = item.avg_cost * item.shares
            gain = current_value - cost_basis
            gain_percent = (gain / cost_basis * 100) if cost_basis > 0 else 0
            
            positions.append({
                "ticker": item.ticker,
                "shares": item.shares,
                "avg_cost": item.avg_cost,
                "current_price": current_price,
                "current_value": current_value,
                "cost_basis": cost_basis,
                "gain": gain,
                "gain_percent": gain_percent
            })
            
            total_value += current_value
            total_cost += cost_basis
            
        except Exception as e:
            print(f"Error fetching price for {item.ticker}: {e}")
            # Use avg_cost as fallback
            positions.append({
                "ticker": item.ticker,
                "shares": item.shares,
                "avg_cost": item.avg_cost,
                "current_price": item.avg_cost,
                "current_value": item.avg_cost * item.shares,
                "cost_basis": item.avg_cost * item.shares,
                "gain": 0,
                "gain_percent": 0
            })
            total_value += item.avg_cost * item.shares
            total_cost += item.avg_cost * item.shares
    
    total_gain = total_value - total_cost
    total_gain_percent = (total_gain / total_cost * 100) if total_cost > 0 else 0
    
    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_gain": total_gain,
        "total_gain_percent": total_gain_percent,
        "positions": positions
    }


# ============================================================================
# RECOMMENDATIONS ENDPOINTS
# ============================================================================

@app.get("/api/recommendations", response_model=List[RecommendationResponse])
def get_recommendations(
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get top recommendations for user based on risk profile"""
    # Get latest recommendations
    latest_date = db.query(DailyRecommendation.date).order_by(DailyRecommendation.date.desc()).first()
    if not latest_date:
        return []
    
    recommendations = db.query(DailyRecommendation).filter(
        DailyRecommendation.date == latest_date[0]
    ).order_by(DailyRecommendation.confidence.desc()).limit(limit).all()
    
    return recommendations


@app.get("/api/recommendations/portfolio", response_model=List[RecommendationResponse])
def get_portfolio_recommendations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get recommendations for stocks in user's portfolio"""
    # Get user's tickers
    portfolio_tickers = db.query(Portfolio.ticker).filter(
        Portfolio.user_id == current_user.id
    ).all()
    tickers = [t[0] for t in portfolio_tickers]
    
    if not tickers:
        return []
    
    # Get latest recommendations for those tickers
    latest_date = db.query(DailyRecommendation.date).order_by(DailyRecommendation.date.desc()).first()
    if not latest_date:
        return []
    
    recommendations = db.query(DailyRecommendation).filter(
        DailyRecommendation.date == latest_date[0],
        DailyRecommendation.ticker.in_(tickers)
    ).all()
    
    return recommendations


# ============================================================================
# NEWS ENDPOINTS
# ============================================================================

@app.get("/api/news")
def get_news(
    ticker: str = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get news articles, optionally filtered by ticker"""
    query = db.query(NewsArticle).order_by(NewsArticle.published_at.desc())
    
    if ticker:
        query = query.filter(NewsArticle.ticker == ticker.upper())
    
    articles = query.limit(limit).all()
    return articles


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/")
def root():
    """API health check"""
    return {
        "status": "online",
        "message": "Stock Predictor API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)