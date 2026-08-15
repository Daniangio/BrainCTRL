# Developing New Components

## Add a Dataset

Create a new `DatasetAdapter` implementation and register it in `bci.registry`. Do not add dataset-specific branches to preprocessing, features, or decoders.

## Add a Decoder

Implement the `Decoder` API. If it can be visualized, expose `transform_latent()` and `diagnostics()` rather than asking GUI code to read private attributes.

## Add Hardware

Prefer hardware software that already publishes LSL. If not, add one `StreamPublisher` that converts the vendor/BrainFlow stream to LSL. Downstream experiment code should remain unchanged.

## Add Features

Implement `FeatureExtractor.transform(trial) -> FeatureRecord`. Preserve provenance and configuration hashes.

## Add GUI Panels

Subscribe to event bus events. Panels should visualize state, not drive experiment transitions or model updates.
