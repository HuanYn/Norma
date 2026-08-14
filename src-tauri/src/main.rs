#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::{
    env,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
};
use tauri::{Manager, State};

const WORKER_URL: &str = "http://127.0.0.1:8765";

#[derive(Default)]
struct WorkerProcess(Mutex<Option<Child>>);

#[derive(Debug, Deserialize)]
struct PythonHealth {
    status: String,
    schema_version: u32,
}

#[derive(Debug, Serialize)]
struct WorkerStatus {
    running: bool,
    healthy: bool,
    url: String,
    message: String,
    schema_version: Option<u32>,
}

#[derive(Debug, Serialize)]
struct AlbumIndexRequest<'a> {
    folder: &'a str,
}

#[derive(Debug, Serialize)]
struct AlbumSearchRequest<'a> {
    album_id: &'a str,
    query: &'a str,
    limit: u32,
}

#[tauri::command]
async fn worker_health() -> WorkerStatus {
    match reqwest::Client::new()
        .get(format!("{WORKER_URL}/health"))
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => {
            match response.json::<PythonHealth>().await {
                Ok(health) => WorkerStatus {
                    running: true,
                    healthy: health.status == "ok",
                    url: WORKER_URL.to_string(),
                    message: "Python AI worker and SQLite are ready".to_string(),
                    schema_version: Some(health.schema_version),
                },
                Err(error) => unavailable(format!("Invalid worker response: {error}")),
            }
        }
        Ok(response) => unavailable(format!("Worker returned HTTP {}", response.status())),
        Err(error) => unavailable(format!("Worker connection failed: {error}")),
    }
}

#[tauri::command]
fn pick_photo_folder() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("Choose a JPG photo folder")
        .pick_folder()
        .map(|path| path.to_string_lossy().to_string())
}

#[tauri::command]
async fn index_album(folder: String) -> Result<serde_json::Value, String> {
    let response = reqwest::Client::new()
        .post(format!("{WORKER_URL}/albums/index"))
        .json(&AlbumIndexRequest { folder: &folder })
        .send()
        .await
        .map_err(|error| format!("Unable to contact AI worker: {error}"))?;
    let status = response.status();
    let payload = response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("Invalid indexing response: {error}"))?;
    if !status.is_success() {
        return Err(payload
            .get("detail")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Album indexing failed")
            .to_string());
    }
    Ok(payload)
}

#[tauri::command]
async fn embed_album(album_id: String) -> Result<serde_json::Value, String> {
    post_worker_json(
        format!("{WORKER_URL}/albums/{album_id}/embed"),
        None::<&serde_json::Value>,
        "Album embedding failed",
    )
    .await
}

#[tauri::command]
async fn search_album(
    album_id: String,
    query: String,
    limit: u32,
) -> Result<serde_json::Value, String> {
    post_worker_json(
        format!("{WORKER_URL}/albums/search"),
        Some(&AlbumSearchRequest {
            album_id: &album_id,
            query: &query,
            limit,
        }),
        "Album search failed",
    )
    .await
}

async fn post_worker_json<T: Serialize + ?Sized>(
    url: String,
    body: Option<&T>,
    fallback_error: &str,
) -> Result<serde_json::Value, String> {
    let request = reqwest::Client::new().post(url);
    let response = match body {
        Some(value) => request.json(value),
        None => request,
    }
    .send()
    .await
    .map_err(|error| format!("Unable to contact AI worker: {error}"))?;
    let status = response.status();
    let payload = response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("Invalid worker response: {error}"))?;
    if !status.is_success() {
        return Err(payload
            .get("detail")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(fallback_error)
            .to_string());
    }
    Ok(payload)
}

fn unavailable(message: String) -> WorkerStatus {
    WorkerStatus {
        running: false,
        healthy: false,
        url: WORKER_URL.to_string(),
        message,
        schema_version: None,
    }
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri must be inside project root")
        .to_path_buf()
}

fn spawn_worker() -> Result<Child, String> {
    let executable = env::var("NORMA_PYTHON").unwrap_or_else(|_| "python".to_string());
    Command::new(executable)
        .args([
            "-m",
            "uvicorn",
            "ai.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
        ])
        .current_dir(project_root())
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("Unable to start Python worker: {error}"))
}

fn main() {
    tauri::Builder::default()
        .manage(WorkerProcess::default())
        .setup(|app| {
            let state = app.state::<WorkerProcess>();
            let mut process = state
                .0
                .lock()
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            *process = Some(spawn_worker().map_err(std::io::Error::other)?);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            worker_health,
            pick_photo_folder,
            index_album,
            embed_album,
            search_album
        ])
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::Destroyed) {
                if let Ok(mut process) = window.state::<WorkerProcess>().0.lock() {
                    if let Some(mut child) = process.take() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run Norma desktop");
}
