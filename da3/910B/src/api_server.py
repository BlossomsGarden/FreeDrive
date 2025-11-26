"""
API server for pose estimation evaluation.
Run with: python api_server.py --port 34567
"""
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from infer_demo import evaluate_pose_estimation
import uvicorn

app = FastAPI(title="Pose Estimation Evaluation API")

class EvaluationRequest(BaseModel):
    video_path: str
    cam_id: str
    gt_json_path: str

class EvaluationResponse(BaseModel):
    rot_err_mean: float
    trans_err_mean: float

@app.post('/evaluate_pose', response_model=EvaluationResponse)
async def evaluate_pose(request: EvaluationRequest):
    """
    Evaluate camera pose estimation accuracy.
    
    Request body (JSON):
    {
        "video_path": "path/to/video.mp4",
        "cam_id": "cam02",
        "gt_json_path": "path/to/camera_extrinsics.json"
    }
    
    Response (JSON):
    {
        "rot_err_mean": 0.001234,
        "trans_err_mean": 0.056789
    }
    """
    try:
        # Call evaluation function
        rot_err_mean, trans_err_mean = evaluate_pose_estimation(
            video_path=request.video_path,
            cam_id=request.cam_id,
            gt_json_path=request.gt_json_path
        )
        
        # Return results
        return EvaluationResponse(
            rot_err_mean=float(rot_err_mean),
            trans_err_mean=float(trans_err_mean)
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get('/health')
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose Estimation Evaluation API Server')
    parser.add_argument('--port', type=int, default=34567, help='Port to run the server on (default: 34567)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    print(f"Starting API server on {args.host}:{args.port}")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print(f"API endpoint: http://{args.host}:{args.port}/evaluate_pose")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(app, host=args.host, port=args.port)

