# Changelog

## [0.1.39](https://github.com/tenfourty/kbx/compare/kbx-v0.1.38...kbx-v0.1.39) (2026-03-09)


### Bug Fixes

* make test suite fully offline — no huggingface.co network calls ([ed52cdb](https://github.com/tenfourty/kbx/commit/ed52cdb531fce8737dfbae01a124b562b764e5fa))

## [0.1.38](https://github.com/tenfourty/kbx/compare/kbx-v0.1.37...kbx-v0.1.38) (2026-03-09)


### Bug Fixes

* strip SOCKS proxy env vars in test conftest ([d7f1bb0](https://github.com/tenfourty/kbx/commit/d7f1bb0a093ebcc72dee387bde5a0312ab5d3eca))

## [0.1.37](https://github.com/tenfourty/kbx/compare/kbx-v0.1.36...kbx-v0.1.37) (2026-03-09)


### Bug Fixes

* guard MLX import against Python 3.14+ segfault ([f4b71cb](https://github.com/tenfourty/kbx/commit/f4b71cb8b188b99ab98d4520a43f9c8d500c1270))

## [0.1.36](https://github.com/tenfourty/kbx/compare/kbx-v0.1.35...kbx-v0.1.36) (2026-03-08)


### Features

* replace Node.js prosemirror-to-ydoc with pure-Python pycrdt ([f8047bf](https://github.com/tenfourty/kbx/commit/f8047bf5b00a4268ae2e11055993a718127eebff))


### Bug Fixes

* add try/except for graceful degradation in ydoc conversion ([e02f706](https://github.com/tenfourty/kbx/commit/e02f706c7288e1ce7606e23a5e82d8065859ed0b))

## [0.1.35](https://github.com/tenfourty/kbx/compare/kbx-v0.1.34...kbx-v0.1.35) (2026-03-08)


### Features

* auto-install prosemirror Node deps on first Granola push-back ([42a41cd](https://github.com/tenfourty/kbx/commit/42a41cd3c96b63207041073addbe112b3ed8a83a))

## [0.1.34](https://github.com/tenfourty/kbx/compare/kbx-v0.1.33...kbx-v0.1.34) (2026-03-08)


### Features

* bundle prosemirror-to-ydoc.mjs inside kbx package ([797f733](https://github.com/tenfourty/kbx/commit/797f733d46cc1b7d38deb88361120c2589087ef7))

## [0.1.33](https://github.com/tenfourty/kbx/compare/kbx-v0.1.32...kbx-v0.1.33) (2026-03-06)


### Bug Fixes

* use calendar event date for sync directory placement + multi-recording variant naming ([f4357df](https://github.com/tenfourty/kbx/commit/f4357dfce89698806b442f666afb5b3b93d74cb0))

## [0.1.32](https://github.com/tenfourty/kbx/compare/kbx-v0.1.31...kbx-v0.1.32) (2026-03-06)


### Features

* make Notion space_id configurable via kbx.toml ([25e3af9](https://github.com/tenfourty/kbx/commit/25e3af98aeae8a0348c37f209aff5118f33899ac))


### Bug Fixes

* resolve relative date shorthand in sync commands ([9f95ba6](https://github.com/tenfourty/kbx/commit/9f95ba60f91656002c14a9e39fa47d78121c6755))

## [0.1.31](https://github.com/tenfourty/kbx/compare/kbx-v0.1.30...kbx-v0.1.31) (2026-03-06)


### Features

* add sources: schema, CLI, and display for project entities ([92e64aa](https://github.com/tenfourty/kbx/commit/92e64aa6f2929b492d263292dc98e13ac08d999b))
* entity linking via source IDs (Phase 4) ([25addde](https://github.com/tenfourty/kbx/commit/25addde3000477962c79ebbdc531d6e7cdf3cf54))


### Documentation

* update CLAUDE.md and docs for source linking and matching.py ([2420a08](https://github.com/tenfourty/kbx/commit/2420a08b891a7dc3b5268f6f4c3587d6e6da1432))

## [0.1.30](https://github.com/tenfourty/kbx/compare/kbx-v0.1.29...kbx-v0.1.30) (2026-03-06)


### Features

* extract task-matching into matching.py with word-boundary fix ([8ecd25a](https://github.com/tenfourty/kbx/commit/8ecd25a963fbf9ecc4e4b8201cffec9d6fd1c8a2))

## [0.1.29](https://github.com/tenfourty/kbx/compare/kbx-v0.1.28...kbx-v0.1.29) (2026-03-06)


### Features

* improve Granola push — ydoc CRDT support, idempotency, formatting ([e1870d5](https://github.com/tenfourty/kbx/commit/e1870d52f99c8b5e4ab47852732f7be5bbd7ddf3))

## [0.1.28](https://github.com/tenfourty/kbx/compare/kbx-v0.1.27...kbx-v0.1.28) (2026-03-03)


### Features

* dynamic staleness detection for all memory subdirectories ([853f0ce](https://github.com/tenfourty/kbx/commit/853f0ceea25f903bc7b398785dfbee098edb2c53))

## [0.1.27](https://github.com/tenfourty/kbx/compare/kbx-v0.1.26...kbx-v0.1.27) (2026-03-03)


### Documentation

* document write-through principle and staleness gap ([3874066](https://github.com/tenfourty/kbx/commit/38740669b1b05d726efb4e9722d2ce12fbf97f92))

## [0.1.26](https://github.com/tenfourty/kbx/compare/kbx-v0.1.25...kbx-v0.1.26) (2026-03-03)


### Features

* index prep and debrief meeting files ([d22fd38](https://github.com/tenfourty/kbx/commit/d22fd38530f57a9f0d6304e6aaa6afb0b8a37eee))

## [0.1.25](https://github.com/tenfourty/kbx/compare/kbx-v0.1.24...kbx-v0.1.25) (2026-03-03)


### Features

* migrate usage command into enhanced --help output ([74fe22b](https://github.com/tenfourty/kbx/commit/74fe22bb437bf4e712cd0f79e1b05e2f1471da6b))

## [0.1.24](https://github.com/tenfourty/kbx/compare/kbx-v0.1.23...kbx-v0.1.24) (2026-03-03)


### Bug Fixes

* correct module — 5 code review bugs ([60ec3de](https://github.com/tenfourty/kbx/commit/60ec3de52c31a7b35bc1852b7f5caca85dfdd79f))

## [0.1.23](https://github.com/tenfourty/kbx/compare/kbx-v0.1.22...kbx-v0.1.23) (2026-03-03)


### Features

* add CLI tests and audit log for kbx correct ([c5346e5](https://github.com/tenfourty/kbx/commit/c5346e5e1076aecf8bbf76baf3c9b419cdedf417))

## [0.1.22](https://github.com/tenfourty/kbx/compare/kbx-v0.1.21...kbx-v0.1.22) (2026-03-03)


### Features

* add `kbx correct` command for find-and-replace across memory files ([cd363d3](https://github.com/tenfourty/kbx/commit/cd363d3427358932d4d6a9e80bb0253e8918c4b9))

## [0.1.21](https://github.com/tenfourty/kbx/compare/kbx-v0.1.20...kbx-v0.1.21) (2026-03-03)


### Features

* add --merge-chunks flag to concatenate per-doc chunks when deduping ([6671cc4](https://github.com/tenfourty/kbx/commit/6671cc43d70affa33ea402101abad81aa1ce2010))

## [0.1.20](https://github.com/tenfourty/kbx/compare/kbx-v0.1.19...kbx-v0.1.20) (2026-03-03)


### Bug Fixes

* use &lt;mark&gt; for search highlighting and suppress stderr in JSON mode ([14e27d8](https://github.com/tenfourty/kbx/commit/14e27d837bca862265796b8927b8882159caeda3))

## [0.1.19](https://github.com/tenfourty/kbx/compare/kbx-v0.1.18...kbx-v0.1.19) (2026-03-03)


### Features

* add highlight param to search for HTML snippet opt-in ([7b77329](https://github.com/tenfourty/kbx/commit/7b77329ac64e12c3fb0cd7d04e32e557023dc1c5))

## [0.1.18](https://github.com/tenfourty/kbx/compare/kbx-v0.1.17...kbx-v0.1.18) (2026-03-03)


### Features

* add --full-chunks, --snippet-chars, and strip HTML from JSON snippets ([ec39873](https://github.com/tenfourty/kbx/commit/ec398730f2360c4bd0b81565f93f27b76da094ad))

## [0.1.17](https://github.com/tenfourty/kbx/compare/kbx-v0.1.16...kbx-v0.1.17) (2026-03-03)


### Performance Improvements

* eliminate redundant API calls in granola edit flow ([1b36d18](https://github.com/tenfourty/kbx/commit/1b36d186cc754fc46b727fd0985a9dadd30fe2d5))

## [0.1.16](https://github.com/tenfourty/kbx/compare/kbx-v0.1.15...kbx-v0.1.16) (2026-03-02)


### Features

* add --transcript flag to granola view command ([7e58ef5](https://github.com/tenfourty/kbx/commit/7e58ef5d1e9df7e2efd1b2faedfa8c9f47dd3f65))

## [0.1.15](https://github.com/tenfourty/kbx/compare/kbx-v0.1.14...kbx-v0.1.15) (2026-03-02)


### Features

* **granola:** add view and edit commands for meeting notes ([0bb5bd5](https://github.com/tenfourty/kbx/commit/0bb5bd5e5b91f7d71aeabce00d880b120a127afb))

## [0.1.14](https://github.com/tenfourty/kbx/compare/kbx-v0.1.13...kbx-v0.1.14) (2026-03-02)


### Documentation

* **usage:** document Granola sync file types in kbx usage ([3874867](https://github.com/tenfourty/kbx/commit/38748670e7b022148efbc67f7b4a4f60325648a2))

## [0.1.13](https://github.com/tenfourty/kbx/compare/kbx-v0.1.12...kbx-v0.1.13) (2026-03-02)


### Bug Fixes

* **meetings:** include .ai-summary.md in meetings walker ([f13f43b](https://github.com/tenfourty/kbx/commit/f13f43b379852d41896c9264fa39d2d4dbb9b1ae))

## [0.1.12](https://github.com/tenfourty/kbx/compare/kbx-v0.1.11...kbx-v0.1.12) (2026-03-02)


### Features

* **granola:** add write API — push prep notes into Granola documents ([3599df6](https://github.com/tenfourty/kbx/commit/3599df69caa106edf03a224ed0d124b1bce2efb2))
* **granola:** auto-create doc when push target not found ([f8bfb2b](https://github.com/tenfourty/kbx/commit/f8bfb2b46b525935014c003497f395e925643830))
* **granola:** sync AI summary as .granola.summary.md ([6d20c05](https://github.com/tenfourty/kbx/commit/6d20c057a17aac130473eb7620ec74fbaa44294e))


### Bug Fixes

* **granola:** interpret \n escapes in --notes, clarify --create limits ([5096773](https://github.com/tenfourty/kbx/commit/5096773d39dc8b973ffd7177f6fa9f6e88f04ec2))
* **granola:** preserve document updated_at to avoid timeline shift ([6eb7b8f](https://github.com/tenfourty/kbx/commit/6eb7b8f10e3a03cb45efbbf9a56f37275bd2b484))
* **sync:** add progress logging to Notion page-filtering loop ([d728635](https://github.com/tenfourty/kbx/commit/d728635206e42ca96b519ddde4ba5200fe9de42d))


### Performance Improvements

* **granola:** early termination in list_documents + tighter date window ([10e4a6f](https://github.com/tenfourty/kbx/commit/10e4a6f271eb6504edba93ce386ff88f8c9e7b4d))
* **sync:** fix Notion sync performance — 65x faster for incremental runs ([0d0045e](https://github.com/tenfourty/kbx/commit/0d0045eb50e487d278c6cc73d923c9dd57cc09d6))


### Documentation

* update Granola write API design for auto-create behavior ([63e99c1](https://github.com/tenfourty/kbx/commit/63e99c1588af771cf9e3ba9f63266cb428e3cb7f))
* update Granola write API design for calendar UID matching ([375dd42](https://github.com/tenfourty/kbx/commit/375dd42d6e56c87b1cb0e1942d773e68a0a0fa5c))

## [0.1.11](https://github.com/tenfourty/kbx/compare/kbx-v0.1.10...kbx-v0.1.11) (2026-03-01)


### Features

* preserve list metadata in entity parser ([3a7e58d](https://github.com/tenfourty/kbx/commit/3a7e58d0a310cd5cc471a746fc27140eb380ad3e))

## [0.1.10](https://github.com/tenfourty/kbx/compare/kbx-v0.1.9...kbx-v0.1.10) (2026-03-01)


### Features

* add batch wikilinks script for memory files ([6be8296](https://github.com/tenfourty/kbx/commit/6be8296c4fe951eb71bd852ba898dc9a4f62693d))
* add strip_wikilinks() utility for [[wikilink]] removal ([388194a](https://github.com/tenfourty/kbx/commit/388194af8ca1c48089286d6c177c46699edad3a6))
* add YAML frontmatter migration script ([30ea238](https://github.com/tenfourty/kbx/commit/30ea23862a126a232837a3bebe58a07ecea6313b))
* **entities:** parse YAML frontmatter with fallback to legacy format ([0ea1b18](https://github.com/tenfourty/kbx/commit/0ea1b18553ea9ba033c6f80e8514e4b1051628d8))
* YAML frontmatter + wikilinks for entity files (Tasks 2-5, 7) ([50b6148](https://github.com/tenfourty/kbx/commit/50b6148d05d19acd33cfaeea9cef6d7f63d44529))


### Bug Fixes

* **entities:** strip wikilinks from team names parsed from company.md ([7e2d658](https://github.com/tenfourty/kbx/commit/7e2d658173a63a8002f71a850f3f9bc90eaf0976))


### Documentation

* add YAML frontmatter + wikilinks design doc ([76d150d](https://github.com/tenfourty/kbx/commit/76d150da3ff9aa1bbf3c1185bd1158d26f9c0a86))
* add YAML frontmatter + wikilinks implementation plan ([31b15af](https://github.com/tenfourty/kbx/commit/31b15af76558439665f264dc65def2b81718c134))

## [0.1.9](https://github.com/tenfourty/kbx/compare/kbx-v0.1.8...kbx-v0.1.9) (2026-03-01)


### Features

* **api:** add LanceDB cleanup and optional re-embedding to reindex helper ([430cfcc](https://github.com/tenfourty/kbx/commit/430cfccf3568263e85e31bc805283be4ce73370e))
* **api:** add post-mutation reindexing for memory file mutations ([e6d9442](https://github.com/tenfourty/kbx/commit/e6d9442640748cbd7af950cc3d77c1c7b2c8f194))
* **api:** expose dedupe, fts_weight, vector_weight in KnowledgeBase.search() ([8eed97a](https://github.com/tenfourty/kbx/commit/8eed97acee8471b6703d6948f2eb7da49f529b89))
* **kbx:** add 'entity stale' command for freshness detection ([20eeec6](https://github.com/tenfourty/kbx/commit/20eeec6002b6abcfa05d26c973fd311ff9f5760f))
* **kbx:** add entity freshness columns (migration 008) ([477b396](https://github.com/tenfourty/kbx/commit/477b396cef8042d73659d021326f8a4838cec77a))
* **kbx:** add entity_stale MCP tool and freshness fields to person_find ([2cb9b83](https://github.com/tenfourty/kbx/commit/2cb9b83cd416bf1f2a39b6bde30402a3c810aa6e))
* **kbx:** add missing CRUD operations — note delete, glossary edit, fact delete/edit ([f654d04](https://github.com/tenfourty/kbx/commit/f654d04343f952e6f16b08a8662cdd3b6567e1b7))
* **kbx:** backfill entity freshness from existing data (migration 009) ([f67b898](https://github.com/tenfourty/kbx/commit/f67b89889a94077da204bc7775513611ddbba14d))
* **kbx:** FTS5 search quick wins — heading boost, AND variant, prefix search ([df0bf82](https://github.com/tenfourty/kbx/commit/df0bf825e01f301ba548054e06f1989890ee194e))
* **kbx:** set updated_at on entity profile edits ([b34fdc5](https://github.com/tenfourty/kbx/commit/b34fdc5d2e1a481282931c45cec613f3da9e447b))
* **kbx:** show freshness indicators in context output ([2239bdd](https://github.com/tenfourty/kbx/commit/2239bdde2394a6f3a716e803142a24377597be2d))
* **kbx:** update last_mentioned_at on entity mention linking ([caaee1e](https://github.com/tenfourty/kbx/commit/caaee1e4991f7810a2ba0a48e97f6b89a792bcef))
* **search:** add configurable weights, dedup, glossary expansion, entity boost, highlighting ([39f65a1](https://github.com/tenfourty/kbx/commit/39f65a17ef0e636f86bae7096bb2e711b0d447bf))
* **search:** add FTS5 UPDATE trigger for chunk content sync ([132d1b0](https://github.com/tenfourty/kbx/commit/132d1b08f1f26893cb1c24d87ccd9b6658758386))
* **search:** center snippet window on first query match ([f59a9a9](https://github.com/tenfourty/kbx/commit/f59a9a97ad44bc585606586ff9ca3f6c873d11bb))
* **search:** separate metadata prefix from chunk content to reduce FTS noise ([2aa2164](https://github.com/tenfourty/kbx/commit/2aa216464b8c18082f2847ed12ceb052c1e23d3e))


### Bug Fixes

* **kbx:** commit DDL migrations before dependent backfills ([08efccf](https://github.com/tenfourty/kbx/commit/08efccf1404bb1b761fdf18af1d0b00ea5530431))
* **kbx:** set updated_at on entity creation, validate --days ([4280c87](https://github.com/tenfourty/kbx/commit/4280c87cb4c76981f90c19e5df0d5d668e0e1f1c))

## [0.1.8](https://github.com/tenfourty/kbx/compare/kbx-v0.1.7...kbx-v0.1.8) (2026-02-25)


### Features

* **entities:** accent-insensitive find + ASCII-only file slugs ([#18](https://github.com/tenfourty/kbx/issues/18)) ([3138eb9](https://github.com/tenfourty/kbx/commit/3138eb98cbba4b100b362ca88445978023ae0704))

## [0.1.7](https://github.com/tenfourty/kbx/compare/kbx-v0.1.6...kbx-v0.1.7) (2026-02-25)


### Features

* **api:** add batch get_fact_counts() method ([#16](https://github.com/tenfourty/kbx/issues/16)) ([19e061e](https://github.com/tenfourty/kbx/commit/19e061e888b8f2417a9ab91afcc6e01d60f56164))
* **api:** add batch get_fact_counts() method ([#16](https://github.com/tenfourty/kbx/issues/16)) ([81252b0](https://github.com/tenfourty/kbx/commit/81252b07a0dc8eada9c74f135e1b75fb4bb92c46))
* **api:** add entity list, get, find operations ([#5](https://github.com/tenfourty/kbx/issues/5)) ([57e619d](https://github.com/tenfourty/kbx/commit/57e619d3e4d7cfb2bfe912e0b62c7b7981248eb3))
* **api:** add entity timeline, pin toggles, document operations ([#5](https://github.com/tenfourty/kbx/issues/5)) ([28d7d2d](https://github.com/tenfourty/kbx/commit/28d7d2deb5ebd9cf36e2ec679b00d1c7643aad46))
* **api:** add KnowledgeBase class with constructor and lifecycle ([#5](https://github.com/tenfourty/kbx/issues/5)) ([dd664d1](https://github.com/tenfourty/kbx/commit/dd664d13c6d5a1a79f59394a1001f2144e2ea842))
* **api:** add memory ops, glossary, search, context, index ([#5](https://github.com/tenfourty/kbx/issues/5)) ([83c9d8f](https://github.com/tenfourty/kbx/commit/83c9d8f3b1cab04c1122c965c4053695a13d4327))
* **api:** add warm_embedder() method to skip auto_reindex ([#17](https://github.com/tenfourty/kbx/issues/17)) ([1546add](https://github.com/tenfourty/kbx/commit/1546add39037c0ba4539d94dc3f2c30547a17472))
* **api:** add warm_embedder() method to skip auto_reindex ([#17](https://github.com/tenfourty/kbx/issues/17)) ([d8a662e](https://github.com/tenfourty/kbx/commit/d8a662e99d6df63ca01c59e3504b87a4f1f7921d))
* **cli:** add --plain flag to view command for raw content output ([2a8a255](https://github.com/tenfourty/kbx/commit/2a8a255502f2541322f0e56d5bf83b44096f16c0))
* **cli:** add --plain flag to view command for raw content output ([#11](https://github.com/tenfourty/kbx/issues/11)) ([ed07d4e](https://github.com/tenfourty/kbx/commit/ed07d4e9178f398453d1f14a2e9a75482c571366))
* **cli:** add note edit command for updating existing memory notes ([bdbc354](https://github.com/tenfourty/kbx/commit/bdbc354556df96ee3378dc9019802d36966a6b1e))
* **cli:** add note edit command for updating existing memory notes ([#12](https://github.com/tenfourty/kbx/issues/12)) ([eb84b47](https://github.com/tenfourty/kbx/commit/eb84b47ce98d62d6305678ad18e23e2a714c0e91))
* **context:** add _format_person_pinned and _format_person_key formatters ([9a873fd](https://github.com/tenfourty/kbx/commit/9a873fde9272df740c48e27cf1a8f7f32ce7347e))
* **context:** add _get_fact_counts and _is_key_person helpers ([1d08129](https://github.com/tenfourty/kbx/commit/1d08129644674a5db8e7ecb4221ba0c3a0f86541))
* **context:** split people into pinned/key tiers in generate_context ([2056590](https://github.com/tenfourty/kbx/commit/205659081a0210c8b31390615891c3943ee7681c))
* **entities:** support custom metadata via --meta flag ([#15](https://github.com/tenfourty/kbx/issues/15)) ([cc28cb9](https://github.com/tenfourty/kbx/commit/cc28cb996696830bf71e984acd1a9bbb22c73006))
* **types:** add API response models for KnowledgeBase class ([#5](https://github.com/tenfourty/kbx/issues/5)) ([f2b9ae6](https://github.com/tenfourty/kbx/commit/f2b9ae65fe474ddb52b05be7f41899832c5c3e52))


### Bug Fixes

* **api:** address code review issues for KnowledgeBase class ([#5](https://github.com/tenfourty/kbx/issues/5)) ([5e9c0ec](https://github.com/tenfourty/kbx/commit/5e9c0ec4979088f2a011723901a9c62d0603a217))
* **entities:** improve entity linking for short names and title matching ([#6](https://github.com/tenfourty/kbx/issues/6)) ([48ca855](https://github.com/tenfourty/kbx/commit/48ca8555a6b195ab9fe5fd9d21193d84798d3369))
* **entities:** lower short-name threshold and add title substring matching ([09d4fea](https://github.com/tenfourty/kbx/commit/09d4fea42ea75947a42dabedf1a8e67fc01e9310))
* **sync:** fetch Notion AI summary for meeting notes ([#14](https://github.com/tenfourty/kbx/issues/14)) ([2e6bf2c](https://github.com/tenfourty/kbx/commit/2e6bf2c88a117fd3fbbaedd41969ad721bb840cd))


### Documentation

* add design for enhanced context people tiers ([076fd03](https://github.com/tenfourty/kbx/commit/076fd0388979af066d2272e0d0fa389bbd94099a))
* add KnowledgeBase API implementation plan ([#5](https://github.com/tenfourty/kbx/issues/5)) ([655ff50](https://github.com/tenfourty/kbx/commit/655ff50e3ae174aae0c4aa12a225e84ed22d0085))
* add Python API module design for KnowledgeBase class ([#5](https://github.com/tenfourty/kbx/issues/5)) ([92d0009](https://github.com/tenfourty/kbx/commit/92d0009692eb3cbd5931bba774751dd9d491a7bc))
* add Python API to architecture and testing docs ([#5](https://github.com/tenfourty/kbx/issues/5)) ([342d89b](https://github.com/tenfourty/kbx/commit/342d89bb1c39780bbf69b3d0312f4de46bd4bc68))
* **cli:** add Python API section to usage output ([#5](https://github.com/tenfourty/kbx/issues/5)) ([b1046c5](https://github.com/tenfourty/kbx/commit/b1046c557ec845ba9736263e0000258cc2345b3c))

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
