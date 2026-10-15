import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  Activity,
  BadgeCheck,
  BrainCircuit,
  CircleHelp,
  Contrast,
  Download,
  Expand,
  FileImage,
  Grid2X2,
  Loader2,
  Maximize2,
  Minus,
  MousePointer2,
  Plus,
  RotateCcw,
  Scan,
  Settings2,
  ShieldCheck,
  Sparkles,
  SunMedium,
  Upload,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

interface ProcessingResponse {
  status: string;

  anatomy: string;
  confidence: number;
  view: string;
  model: string;

  before: {
    snr_db: number;
    noise: number;
    signal: number;
    contrast: number;
    dynamic_range: number;
    entropy: number;
  };

  after: {
    snr_db: number;
    noise: number;
    signal: number;
    contrast: number;
    dynamic_range: number;
    entropy: number;
  };

  parameters: {
    bilateral_sigma_color: number;
    clahe_clip_limit: number;
    gamma: number;
    sharpen_amount: number;
  };

  dicom_url: string;

  original_preview_url: string;
  enhanced_preview_url: string;
}

type ViewerMode = "original" | "enhanced";

interface ViewerTransform {
  scale: number;
  x: number;
  y: number;
}

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const viewerRef = useRef<HTMLDivElement>(null);

  const [selectedFile, setSelectedFile] =
    useState<File | null>(null);

  const [width, setWidth] = useState("1024");
  const [height, setHeight] = useState("1024");
  const [bitDepth, setBitDepth] = useState("16");
  const [endian, setEndian] = useState("little");

  const [processing, setProcessing] =
    useState(false);

  const [result, setResult] =
    useState<ProcessingResponse | null>(null);

  const [error, setError] = useState("");

  const [viewerMode, setViewerMode] =
    useState<ViewerMode>("enhanced");

  const [zoom, setZoom] = useState(1);

  const [brightness, setBrightness] =
    useState(100);

  const [contrast, setContrast] =
    useState(100);

  const [inverted, setInverted] =
    useState(false);

  const [isFullscreen, setIsFullscreen] =
    useState(false);

  const [showGrid, setShowGrid] =
    useState(false);

  const [isPanning, setIsPanning] =
    useState(false);

  const [transform, setTransform] =
    useState<ViewerTransform>({
      scale: 1,
      x: 0,
      y: 0,
    });

  const [dragStart, setDragStart] =
    useState({
      x: 0,
      y: 0,
      originX: 0,
      originY: 0,
    });

  const [imageLoaded, setImageLoaded] =
    useState(false);

  const [imageError, setImageError] =
    useState(false);

  const handleFileSelect = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);
    setResult(null);
    setError("");

    setViewerMode("enhanced");
    resetViewer();

    setImageLoaded(false);
    setImageError(false);
  };

  const handleProcess = async () => {
    if (!selectedFile) {
      setError(
        "Please select a RAW radiograph first.",
      );
      return;
    }

    setProcessing(true);
    setError("");
    setResult(null);
    setImageLoaded(false);
    setImageError(false);

    try {
      const formData = new FormData();

      formData.append(
        "file",
        selectedFile,
      );

      formData.append(
        "width",
        width,
      );

      formData.append(
        "height",
        height,
      );

      formData.append(
        "bit_depth",
        bitDepth,
      );

      formData.append(
        "signed",
        "false",
      );

      formData.append(
        "endian",
        endian,
      );

      const response = await fetch(
        `${API_BASE_URL}/process`,
        {
          method: "POST",
          body: formData,
        },
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.message ||
            "Radiograph processing failed.",
        );
      }

      setResult(data);

      setViewerMode("enhanced");

      resetViewer();

      setImageLoaded(false);
      setImageError(false);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to connect to the processing engine.";

      setError(message);
    } finally {
      setProcessing(false);
    }
  };

  const handleExport = () => {
    if (!result) {
      return;
    }

    window.open(
      `${API_BASE_URL}${result.dicom_url}`,
      "_blank",
    );
  };

  const resetViewer = useCallback(() => {
    setZoom(1);

    setTransform({
      scale: 1,
      x: 0,
      y: 0,
    });

    setBrightness(100);
    setContrast(100);
    setInverted(false);
  }, []);

  const zoomIn = () => {
    setZoom((current) => {
      const next = Math.min(
        current + 0.25,
        5,
      );

      setTransform((previous) => ({
        ...previous,
        scale: next,
      }));

      return next;
    });
  };

  const zoomOut = () => {
    setZoom((current) => {
      const next = Math.max(
        current - 0.25,
        0.25,
      );

      setTransform((previous) => ({
        ...previous,
        scale: next,
      }));

      return next;
    });
  };

  const fitViewer = () => {
    setZoom(1);

    setTransform({
      scale: 1,
      x: 0,
      y: 0,
    });
  };

  const handleWheel = (
    event: React.WheelEvent<HTMLDivElement>,
  ) => {
    event.preventDefault();

    if (event.deltaY < 0) {
      zoomIn();
    } else {
      zoomOut();
    }
  };

  const handlePointerDown = (
    event: React.PointerEvent<HTMLDivElement>,
  ) => {
    if (
      event.button !== 0 ||
      zoom <= 1
    ) {
      return;
    }

    event.currentTarget.setPointerCapture(
      event.pointerId,
    );

    setIsPanning(true);

    setDragStart({
      x: event.clientX,
      y: event.clientY,
      originX: transform.x,
      originY: transform.y,
    });
  };

  const handlePointerMove = (
    event: React.PointerEvent<HTMLDivElement>,
  ) => {
    if (!isPanning) {
      return;
    }

    const deltaX =
      event.clientX - dragStart.x;

    const deltaY =
      event.clientY - dragStart.y;

    setTransform((previous) => ({
      ...previous,
      x: dragStart.originX + deltaX,
      y: dragStart.originY + deltaY,
    }));
  };

  const stopPanning = () => {
    setIsPanning(false);
  };

  const toggleFullscreen = async () => {
    if (!viewerRef.current) {
      return;
    }

    try {
      if (!document.fullscreenElement) {
        await viewerRef.current.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch {
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(
        Boolean(document.fullscreenElement),
      );
    };

    document.addEventListener(
      "fullscreenchange",
      handleFullscreenChange,
    );

    return () => {
      document.removeEventListener(
        "fullscreenchange",
        handleFullscreenChange,
      );
    };
  }, []);

  useEffect(() => {
    const handleKeyboard = (
      event: KeyboardEvent,
    ) => {
      if (
        event.target instanceof
          HTMLInputElement ||
        event.target instanceof
          HTMLSelectElement ||
        event.target instanceof
          HTMLTextAreaElement
      ) {
        return;
      }

      switch (event.key.toLowerCase()) {
        case "+":
        case "=":
          zoomIn();
          break;

        case "-":
          zoomOut();
          break;

        case "r":
          resetViewer();
          break;

        case "f":
          fitViewer();
          break;

        case "i":
          setInverted(
            (current) => !current,
          );
          break;

        case "o":
          if (result) {
            setViewerMode("original");
            setImageLoaded(false);
          }
          break;

        case "e":
          if (result) {
            setViewerMode("enhanced");
            setImageLoaded(false);
          }
          break;

        case "escape":
          if (document.fullscreenElement) {
            document.exitFullscreen();
          }
          break;

        default:
          break;
      }
    };

    window.addEventListener(
      "keydown",
      handleKeyboard,
    );

    return () => {
      window.removeEventListener(
        "keydown",
        handleKeyboard,
      );
    };
  }, [
    result,
    resetViewer,
    transform,
    isPanning,
    dragStart,
  ]);

  const snrBefore =
    result?.before.snr_db ?? null;

  const snrAfter =
    result?.after.snr_db ?? null;

  const snrImprovement =
    snrBefore !== null &&
    snrAfter !== null
      ? snrAfter - snrBefore
      : null;

  const currentImageUrl = useMemo(() => {
    if (!result) {
      return "";
    }

    const path =
      viewerMode === "original"
        ? result.original_preview_url
        : result.enhanced_preview_url;

    return `${API_BASE_URL}${path}`;
  }, [result, viewerMode]);

  const imageFilter = useMemo(() => {
    return [
      `brightness(${brightness}%)`,
      `contrast(${contrast}%)`,
      inverted
        ? "invert(1)"
        : "invert(0)",
    ].join(" ");
  }, [
    brightness,
    contrast,
    inverted,
  ]);

  return (
    <div className="app-shell">

      {/* ================================================== */}
      {/* TOP BAR */}
      {/* ================================================== */}

      <header className="topbar">

        <div className="brand">

          <div className="brand-mark">
            <Activity size={21} />
          </div>

          <div>
            <div className="brand-name">
              Radiograph AI
            </div>

            <div className="brand-subtitle">
              Adaptive Radiography Processing
            </div>
          </div>

        </div>

        <div className="topbar-center">
          <div className="pipeline-indicator">
            <span className="pipeline-node active" />
            <span className="pipeline-line" />
            <span className="pipeline-node active" />
            <span className="pipeline-line" />
            <span className="pipeline-node active" />
          </div>

          <span>
            RAW · ANALYZE · ENHANCE
          </span>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          <span>
            Processing Engine Online
          </span>
        </div>

      </header>


      {/* ================================================== */}
      {/* WORKSPACE */}
      {/* ================================================== */}

      <main className="workspace">

        {/* ================================================== */}
        {/* LEFT PANEL */}
        {/* ================================================== */}

        <aside className="left-panel">

          <div className="panel-heading">

            <div>
              <span className="eyebrow">
                INPUT
              </span>

              <h2>
                Radiograph
              </h2>
            </div>

            <FileImage size={18} />

          </div>


          {/* Upload */}

          <div className="upload-zone">

            <div className="upload-icon">
              <Upload size={22} />
            </div>

            <h3>
              {selectedFile
                ? selectedFile.name
                : "Upload RAW image"}
            </h3>

            <p>
              {selectedFile
                ? `${(
                    selectedFile.size /
                    (1024 * 1024)
                  ).toFixed(2)} MB selected`
                : "Drop a detector RAW file here or select one from your computer."}
            </p>

            <input
              ref={fileInputRef}
              type="file"
              accept=".raw"
              hidden
              onChange={handleFileSelect}
            />

            <button
              className="secondary-button"
              onClick={() =>
                fileInputRef.current?.click()
              }
            >
              <Upload size={16} />

              {selectedFile
                ? "Change RAW"
                : "Select RAW"}
            </button>

            <span className="file-hint">
              16-bit RAW supported
            </span>

          </div>


          {/* RAW profile */}

          <div className="control-section">

            <div className="section-title">
              <Settings2 size={15} />
              RAW Profile
            </div>

            <div className="field-grid">

              <label>
                Width

                <input
                  value={width}
                  onChange={(event) =>
                    setWidth(
                      event.target.value,
                    )
                  }
                />
              </label>

              <label>
                Height

                <input
                  value={height}
                  onChange={(event) =>
                    setHeight(
                      event.target.value,
                    )
                  }
                />
              </label>

            </div>

            <label className="full-field">
              Bit depth

              <select
                value={bitDepth}
                onChange={(event) =>
                  setBitDepth(
                    event.target.value,
                  )
                }
              >
                <option value="16">
                  16-bit
                </option>

                <option value="14">
                  14-bit
                </option>

                <option value="12">
                  12-bit
                </option>

                <option value="8">
                  8-bit
                </option>
              </select>
            </label>

            <label className="full-field">
              Endianness

              <select
                value={endian}
                onChange={(event) =>
                  setEndian(
                    event.target.value,
                  )
                }
              >
                <option value="little">
                  Little endian
                </option>

                <option value="big">
                  Big endian
                </option>
              </select>
            </label>

          </div>


          {/* Processing */}

          <button
            className="process-button"
            onClick={handleProcess}
            disabled={
              processing ||
              !selectedFile
            }
          >
            {processing ? (
              <>
                <Loader2
                  size={17}
                  className="spin"
                />

                Processing...
              </>
            ) : (
              <>
                <Sparkles size={17} />

                Process Radiograph
              </>
            )}
          </button>


          {error && (
            <div className="error-message">
              {error}
            </div>
          )}


          {/* Processing pipeline card */}

          <div className="mini-pipeline">

            <div className="section-title">
              <BrainCircuit size={15} />
              Processing Pipeline
            </div>

            <div className="pipeline-step">
              <span>01</span>
              RAW ingestion
            </div>

            <div className="pipeline-step">
              <span>02</span>
              Quality analysis
            </div>

            <div className="pipeline-step">
              <span>03</span>
              Anatomy detection
            </div>

            <div className="pipeline-step">
              <span>04</span>
              Adaptive enhancement
            </div>

            <div className="pipeline-step">
              <span>05</span>
              DICOM generation
            </div>

          </div>

        </aside>


        {/* ================================================== */}
        {/* CENTER VIEWER */}
        {/* ================================================== */}

        <section className="viewer-panel">

          <div className="viewer-toolbar">

            <div className="viewer-title">

              <span className="eyebrow">
                IMAGE VIEWER
              </span>

              <h2>
                Study Preview
              </h2>

            </div>


            <div className="viewer-actions">

              <button
                className={
                  viewerMode ===
                  "original"
                    ? "toolbar-button active"
                    : "toolbar-button"
                }
                title="Original image (O)"
                onClick={() => {
                  if (result) {
                    setViewerMode(
                      "original",
                    );
                    setImageLoaded(false);
                  }
                }}
                disabled={!result}
              >
                Original
              </button>

              <button
                className={
                  viewerMode ===
                  "enhanced"
                    ? "toolbar-button active"
                    : "toolbar-button"
                }
                title="Enhanced image (E)"
                onClick={() => {
                  if (result) {
                    setViewerMode(
                      "enhanced",
                    );
                    setImageLoaded(false);
                  }
                }}
                disabled={!result}
              >
                Enhanced
              </button>

              <button
                className="icon-toolbar-button"
                title="Zoom in (+)"
                onClick={zoomIn}
                disabled={!result}
              >
                <ZoomIn size={17} />
              </button>

              <button
                className="icon-toolbar-button"
                title="Zoom out (-)"
                onClick={zoomOut}
                disabled={!result}
              >
                <ZoomOut size={17} />
              </button>

              <button
                className="icon-toolbar-button"
                title="Fit image (F)"
                onClick={fitViewer}
                disabled={!result}
              >
                <Maximize2 size={17} />
              </button>

              <button
                className="icon-toolbar-button"
                title="Reset viewer (R)"
                onClick={resetViewer}
                disabled={!result}
              >
                <RotateCcw size={17} />
              </button>

              <button
                className="icon-toolbar-button"
                title="Fullscreen"
                onClick={toggleFullscreen}
                disabled={!result}
              >
                <Expand size={17} />
              </button>

            </div>

          </div>


          {/* Viewer stage */}

          <div
            ref={viewerRef}
            className={
              isFullscreen
                ? "image-stage fullscreen-stage"
                : "image-stage"
            }
            onWheel={handleWheel}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={stopPanning}
            onPointerCancel={stopPanning}
            onPointerLeave={stopPanning}
          >

            {/* Viewer HUD */}

            {result && (
              <>
                <div className="viewer-hud top-left">

                  <span className="hud-pill">
                    {viewerMode ===
                    "original"
                      ? "ORIGINAL"
                      : "ENHANCED"}
                  </span>

                  <span className="hud-pill subtle">
                    {result.anatomy}
                  </span>

                  <span className="hud-pill subtle">
                    {result.view}
                  </span>

                </div>

                <div className="viewer-hud top-right">

                  <span className="hud-pill">
                    {Math.round(
                      zoom * 100,
                    )}
                    %
                  </span>

                  {inverted && (
                    <span className="hud-pill">
                      INVERTED
                    </span>
                  )}

                </div>
              </>
            )}


            {/* Empty state */}

            {!result && (
              <div className="empty-viewer">

                <div className="empty-icon">
                  <Scan size={34} />
                </div>

                <h3>
                  No radiograph loaded
                </h3>

                <p>
                  Upload a RAW radiograph
                  to begin
                  anatomy-aware
                  processing.
                </p>

                <span>
                  {selectedFile
                    ? selectedFile.name
                    : "Viewer ready"}
                </span>

              </div>
            )}


            {/* Image */}

            {result && (
              <div className="image-canvas">

                {!imageLoaded &&
                  !imageError && (
                    <div className="image-loading">
                      <Loader2
                        size={28}
                        className="spin"
                      />

                      <span>
                        Loading radiograph...
                      </span>
                    </div>
                  )}


                {imageError && (
                  <div className="image-loading">
                    <CircleHelp
                      size={28}
                    />

                    <span>
                      Unable to load preview
                    </span>
                  </div>
                )}


                {!imageError && (
                  <img
                    src={currentImageUrl}
                    alt={`${viewerMode} radiograph`}
                    draggable={false}
                    onLoad={() =>
                      setImageLoaded(true)
                    }
                    onError={() =>
                      setImageError(true)
                    }
                    className={
                      isPanning
                        ? "radiograph-image dragging"
                        : "radiograph-image"
                    }
                    style={{
                      transform: `translate3d(${transform.x}px, ${transform.y}px, 0) scale(${transform.scale})`,
                      filter: imageFilter,
                    }}
                  />
                )}

              </div>
            )}


            {/* Grid */}

            {showGrid && result && (
              <div className="viewer-grid" />
            )}


            {/* Crosshair */}

            {result && (
              <div className="viewer-crosshair">
                <span />
                <span />
              </div>
            )}


            {/* Bottom controls */}

            {result && (
              <div className="viewer-control-dock">

                <button
                  title="Zoom out"
                  onClick={zoomOut}
                >
                  <Minus size={15} />
                </button>

                <div className="zoom-indicator">
                  {Math.round(
                    zoom * 100,
                  )}
                  %
                </div>

                <button
                  title="Zoom in"
                  onClick={zoomIn}
                >
                  <Plus size={15} />
                </button>

                <div className="dock-divider" />

                <button
                  className={
                    inverted
                      ? "dock-active"
                      : ""
                  }
                  title="Invert"
                  onClick={() =>
                    setInverted(
                      (current) =>
                        !current,
                    )
                  }
                >
                  <Contrast size={15} />
                </button>

                <button
                  className={
                    showGrid
                      ? "dock-active"
                      : ""
                  }
                  title="Grid"
                  onClick={() =>
                    setShowGrid(
                      (current) =>
                        !current,
                    )
                  }
                >
                  <Grid2X2 size={15} />
                </button>

                <button
                  title="Reset"
                  onClick={resetViewer}
                >
                  <RotateCcw
                    size={15}
                  />
                </button>

              </div>
            )}


            {/* Footer */}

            <div className="viewer-footer">

              <span>
                {imageLoaded
                  ? `${Math.round(
                      zoom * 100,
                    )}%`
                  : "READY"}
              </span>

              <span>
                MONOCHROME
              </span>

              <span>
                {width} × {height}
              </span>

              {result && (
                <span>
                  {viewerMode ===
                  "original"
                    ? "SOURCE"
                    : "PROCESSED"}
                </span>
              )}

            </div>

          </div>


          {/* Image controls */}

          {result && (
            <div className="image-controls">

              <div className="image-control">

                <SunMedium size={14} />

                <span>
                  Brightness
                </span>

                <input
                  type="range"
                  min="50"
                  max="150"
                  value={brightness}
                  onChange={(event) =>
                    setBrightness(
                      Number(
                        event.target.value,
                      ),
                    )
                  }
                />

                <strong>
                  {brightness}%
                </strong>

              </div>


              <div className="image-control">

                <Contrast size={14} />

                <span>
                  Contrast
                </span>

                <input
                  type="range"
                  min="50"
                  max="180"
                  value={contrast}
                  onChange={(event) =>
                    setContrast(
                      Number(
                        event.target.value,
                      ),
                    )
                  }
                />

                <strong>
                  {contrast}%
                </strong>

              </div>


              <div className="viewer-help">
                <MousePointer2
                  size={13}
                />

                Drag to pan · Scroll
                to zoom · R to reset
              </div>

            </div>
          )}

        </section>


        {/* ================================================== */}
        {/* RIGHT PANEL */}
        {/* ================================================== */}

        <aside className="right-panel">

          <div className="panel-heading">

            <div>

              <span className="eyebrow">
                ANALYSIS
              </span>

              <h2>
                Quality Intelligence
              </h2>

            </div>

            <ShieldCheck size={18} />

          </div>


          {/* Anatomy */}

          <div className="anatomy-card">

            <span className="metric-label">
              DETECTED ANATOMY
            </span>

            <div className="anatomy-value">

              <strong>
                {result?.anatomy ??
                  "—"}
              </strong>

              <span>
                {result
                  ? `${(
                      result.confidence *
                      100
                    ).toFixed(1)}%`
                  : "—"}
              </span>

            </div>

            <div className="confidence-bar">

              <div
                style={{
                  width: result
                    ? `${result.confidence * 100}%`
                    : "0%",
                }}
              />

            </div>

            <span className="view-label">
              View ·{" "}
              {result?.view ?? "—"}
            </span>

          </div>


          {/* Metrics */}

          <div className="metric-grid">

            <div className="metric-card">

              <span>
                SNR BEFORE
              </span>

              <strong>
                {result
                  ? result.before.snr_db.toFixed(
                      2,
                    )
                  : "—"}
              </strong>

              <small>
                dB
              </small>

            </div>


            <div className="metric-card highlight">

              <span>
                SNR AFTER
              </span>

              <strong>
                {result
                  ? result.after.snr_db.toFixed(
                      2,
                    )
                  : "—"}
              </strong>

              <small>
                dB
              </small>

            </div>


            <div className="metric-card">

              <span>
                NOISE
              </span>

              <strong>
                {result
                  ? result.after.noise.toFixed(
                      4,
                    )
                  : "—"}
              </strong>

              <small>
                after
              </small>

            </div>


            <div className="metric-card">

              <span>
                ENTROPY
              </span>

              <strong>
                {result
                  ? result.after.entropy.toFixed(
                      2,
                    )
                  : "—"}
              </strong>

              <small>
                after
              </small>

            </div>

          </div>


          {/* Improvement */}

          <div className="improvement-card">

            <div className="improvement-header">

              <span className="metric-label">
                QUALITY IMPROVEMENT
              </span>

              <BadgeCheck
                size={17}
              />

            </div>

            <div className="improvement-value">

              {snrImprovement !== null
                ? `+${snrImprovement.toFixed(
                    2,
                  )} dB`
                : "—"}

            </div>

            <p>
              Estimated SNR improvement
              after adaptive enhancement.
            </p>

          </div>


          {/* Parameters */}

          <div className="parameters">

            <div className="section-title">
              <Settings2 size={15} />
              Applied Parameters
            </div>


            <div className="parameter-row">

              <span>
                CLAHE
              </span>

              <strong>
                {result
                  ? result.parameters.clahe_clip_limit.toFixed(
                      2,
                    )
                  : "—"}
              </strong>

            </div>


            <div className="parameter-row">

              <span>
                Gamma
              </span>

              <strong>
                {result
                  ? result.parameters.gamma.toFixed(
                      2,
                    )
                  : "—"}
              </strong>

            </div>


            <div className="parameter-row">

              <span>
                Denoise
              </span>

              <strong>
                {result
                  ? result.parameters.bilateral_sigma_color.toFixed(
                      1,
                    )
                  : "—"}
              </strong>

            </div>


            <div className="parameter-row">

              <span>
                Sharpen
              </span>

              <strong>
                {result
                  ? result.parameters.sharpen_amount.toFixed(
                      2,
                    )
                  : "—"}
              </strong>

            </div>

          </div>


          {/* Model */}

          <div className="model-card">

            <div className="model-icon">
              <BrainCircuit
                size={17}
              />
            </div>

            <div>

              <span>
                ANALYSIS MODEL
              </span>

              <strong>
                {result?.model ??
                  "Awaiting study"}
              </strong>

            </div>

          </div>


          {/* Export */}

          <button
            className="export-button"
            disabled={!result}
            onClick={handleExport}
          >
            <Download size={17} />
            Export DICOM
          </button>

        </aside>

      </main>


      {/* ================================================== */}
      {/* FOOTER */}
      {/* ================================================== */}

      <footer className="statusbar">

        <span>
          <span className="status-dot" />
          Research prototype
        </span>

        <span>
          Not clinically validated
          for diagnosis
        </span>

        <span>
          Radiograph AI · v0.1.0
        </span>

      </footer>

    </div>
  );
}

export default App;