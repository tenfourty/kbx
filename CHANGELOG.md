# Changelog

## [0.1.6](https://github.com/tenfourty/kbx/compare/kbx-v0.1.5...kbx-v0.1.6) (2026-02-24)


### Bug Fixes

* **embeddings:** use mx.clear_cache instead of deprecated mx.metal.clear_cache ([548abee](https://github.com/tenfourty/kbx/commit/548abeea1d2ce5aa3ce2057c099f5fb98ecf1bf1))

## [0.1.5](https://github.com/tenfourty/kbx/compare/kbx-v0.1.4...kbx-v0.1.5) (2026-02-24)


### Features

* **sync:** merge Notion sync with emoji-consistent naming ([a9d7677](https://github.com/tenfourty/kbx/commit/a9d767724c349ce3b9b0b0181d815b0913132b93))

## [0.1.4](https://github.com/tenfourty/kbx/compare/kbx-v0.1.3...kbx-v0.1.4) (2026-02-24)


### Features

* **sync:** adopt {uid}_{Title}.granola.{type}.md file naming convention ([adf804d](https://github.com/tenfourty/kbx/commit/adf804defb5f2fcf3cc861fa44f6b2a3a1ce677a))


### Documentation

* add Notion transcript sync design ([b9b695a](https://github.com/tenfourty/kbx/commit/b9b695a704d2f2dbc90d04ddb21d7fa7f12f245b))
* add Notion transcript sync implementation plan ([5596b91](https://github.com/tenfourty/kbx/commit/5596b91df21a0ad920aeada73742b6254646e5c9))

## [0.1.3](https://github.com/tenfourty/kbx/compare/kbx-v0.1.2...kbx-v0.1.3) (2026-02-23)


### Features

* **cli:** add 'kbx me' top-level shortcut ([#3](https://github.com/tenfourty/kbx/issues/3)) ([82045ab](https://github.com/tenfourty/kbx/commit/82045abd745a112cf3aad7ff0b034d475cb73c76))
* **entity-find:** add 'me' shortcut resolving [user] name from config ([#3](https://github.com/tenfourty/kbx/issues/3)) ([a24d8d3](https://github.com/tenfourty/kbx/commit/a24d8d3c74a2730ddf58f31dc6a588cb946cd79e))


### Bug Fixes

* **entity-find:** human-readable table format for entity profiles ([#3](https://github.com/tenfourty/kbx/issues/3)) ([504bae3](https://github.com/tenfourty/kbx/commit/504bae3f52a14034f0bf72280c7e7fa57a09eb16))
* **entity-find:** replace doc dump with compact profile ([#3](https://github.com/tenfourty/kbx/issues/3)) ([ae986d6](https://github.com/tenfourty/kbx/commit/ae986d6d22d2e71a9c268d2070f059fa28b4fc8c))
* **mcp:** update person find to compact output + update usage text ([#3](https://github.com/tenfourty/kbx/issues/3)) ([6b8f0ea](https://github.com/tenfourty/kbx/commit/6b8f0eaf9d327625c68d374d6b30d9ac5cc8b6e6))

## [0.1.2](https://github.com/tenfourty/kbx/compare/kbx-v0.1.1...kbx-v0.1.2) (2026-02-23)


### Bug Fixes

* **embeddings:** use config data dir for model cache ([9875465](https://github.com/tenfourty/kbx/commit/9875465f885fa6a93ed45184a4f96dd2f45132ae))

## [0.1.1](https://github.com/tenfourty/kbx/compare/kbx-v0.1.0...kbx-v0.1.1) (2026-02-23)


### Features

* initial release of kbx — local knowledge base CLI ([d260f66](https://github.com/tenfourty/kbx/commit/d260f66d9dcdb4b876727bd95ec125f6d541a4ee))


### Bug Fixes

* add autouse fixture for fake mlx modules in tests ([3c1dd32](https://github.com/tenfourty/kbx/commit/3c1dd32e7217a0f3cf02871b8bd1433ad10c1ea9))
* MLX mock tests work without mlx installed ([89e12f3](https://github.com/tenfourty/kbx/commit/89e12f3eb64682f8e30e3ecf59128ddd778a4992))
* use NDArray type annotations for numpy compatibility ([4b407ed](https://github.com/tenfourty/kbx/commit/4b407ed9a5895afdef7136fa0456634766d3352b))
