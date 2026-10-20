# Project diagrams

These diagrams describe the two inference paths currently implemented in the
project. They use Mermaid and can be rendered by GitHub, GitLab, and many
Markdown editors.

Native D2 source files and rendered SVGs are also provided:

- [Cellpose pipeline D2 source](cellpose_pipeline.d2) · [SVG](cellpose_pipeline.svg)
- [Ellipse pipeline D2 source](ellipse_pipeline.d2) · [SVG](ellipse_pipeline.svg)
- [Shared components D2 source](shared_components.d2) · [SVG](shared_components.svg)

## Cellpose/ImageJ pipeline

```mermaid
flowchart LR
    A[Saved micrograph open in ImageJ] --> B[Run_AI_Detection.java]
    B --> C[ProjectSettings]
    C --> D[.venv Python]
    D --> E[predict.py]
    E --> F[Cellpose SAM model]
    F --> G[predictions.json]
    G --> H[JsonLoader Gson parser]
    H --> I[RoiCreator]
    I --> J[ImageJ PolygonRoi objects]
    J --> K[RoiManager / Show All]
```

## Experimental ellipse pipeline

```mermaid
flowchart LR
    A[Saved micrograph open in ImageJ] --> B[Run_AI_DetectionEllipse.java]
    B --> C[ProjectSettings]
    C --> D[.venv Python]
    D --> E[predict_ellipse.py]
    E --> F[Grayscale + blur]
    F --> G[Threshold bright/dark regions]
    G --> H[OpenCV contours]
    H --> I[cv2.fitEllipse]
    I --> J{Area, axis ratio,
    axis size, fit error}
    J -->|accepted| K[Ellipse polygons]
    J -->|rejected| L[Discard contour]
    K --> M[ellipse_predictions.json]
    M --> N[JsonLoader + RoiCreator]
    N --> O[RoiManager / Show All]
```

## Shared components

```mermaid
flowchart TB
    S[ProjectSettings] --> P[Python executable]
    S --> W[Project working directory]
    P --> CP[predict.py / predict_ellipse.py]
    CP --> JSON[Prediction JSON schema]
    JSON --> JL[JsonLoader]
    JL --> RC[RoiCreator]
    RC --> RM[ImageJ RoiManager]
```
