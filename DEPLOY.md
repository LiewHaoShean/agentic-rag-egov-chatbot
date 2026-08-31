# Deploying MESRA to a Google Compute Engine VM

Replaces the `cloudflared` development tunnel with a persistent public endpoint.

## Why Compute Engine rather than Cloud Run

Cloud Run provides HTTPS without configuration, which is attractive, but three of
the four services do not fit its execution model. n8n holds workflow state on
disk and must stay resident, the Celery worker is a long-running consumer with no
HTTP surface to scale on, and Redis needs persistence. One small VM running all
four under Docker Compose is simpler and cheaper than splitting them across Cloud
Run, Memorystore and a separate host for n8n.

## What becomes publicly reachable

Only Caddy, on ports 80 and 443. Meta calls the n8n webhook through it over TLS,
and every hop after that stays on the internal Docker network. The API, the
worker and Redis have no published ports at all, which is stricter than the
development setup where the API listened on a host port alongside the tunnel.

```
Meta WhatsApp Cloud API
        │  https://<PUBLIC_HOST>/webhook/...
        ▼
     Caddy  (TLS, only exposed service)
        ▼
      n8n  ──http://api:8000──▶ FastAPI ──▶ Redis ──▶ Celery worker
        ▲                                                    │
        └──────────── outbound callback (internal) ──────────┘
```

## 1. Create the VM

Cloud Console, Compute Engine, Create instance.

| Setting | Value |
|---|---|
| Machine type | `e2-small` (2 vCPU, 2 GB) minimum, `e2-medium` for headroom |
| Region | `asia-southeast1` — same region as `VERTEX_LOCATION`, so model calls stay in-region |
| Boot disk | Debian 12, 20 GB |
| Firewall | tick **Allow HTTP traffic** and **Allow HTTPS traffic** |
| Service account | default Compute Engine account, **Allow full access to all Cloud APIs** |

`e2-micro` has 1 GB and will thrash once n8n, the API, the worker and Redis are
all resident. It is not worth the saving.

The service account scope matters. With it, `GEMINI_BACKEND=vertex` authenticates
through the instance metadata server using Application Default Credentials, so no
service-account key file is needed on the VM and none has to be committed
anywhere. Confirm the Vertex AI API is enabled on the project.

## 2. Reserve a static IP

VPC network, IP addresses, reserve the instance's external address as static.
Without this the address changes on restart and the certificate stops matching.

## 3. Choose the public hostname

**Free** — use `sslip.io`, which resolves any `a-b-c-d.sslip.io` to `a.b.c.d`
with no DNS account. For `34.87.1.2` the hostname is `34-87-1-2.sslip.io`, and
Let's Encrypt issues for it normally.

**Better for a demo** — register a domain and add an A record pointing at the
static IP. A real hostname reads better in a viva than an IP-derived one.

Either way Caddy obtains and renews the certificate automatically.

## 4. Install Docker on the VM

SSH in from the Cloud Console, then:

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
exit                      # log out and back in for the group to apply
```

## 5. Copy the project across

```bash
git clone <your-repo> mesra && cd mesra
```

If the project is not in a remote repository, upload it with the SSH window's
file-transfer option, or `gcloud compute scp --recurse ./fyp VM_NAME:~/mesra`.

## 6. Write the VM's `.env`

Copy your local `.env` across, then change these. Do not commit it.

```dotenv
# Reached by service name on the internal network, not localhost.
REDIS_URL=redis://redis:6379/0

# Vertex in the same region as the VM.
GEMINI_BACKEND=vertex
VERTEX_LOCATION=asia-southeast1
GOOGLE_CLOUD_PROJECT=<your-project-id>

# n8n calls the API by service name; the API calls n8n back the same way.
N8N_CALLBACK_URL=http://n8n:5678/webhook/mesra-reply

# Used by docker-compose and the Caddyfile.
PUBLIC_HOST=34-87-1-2.sslip.io
```

`GOOGLE_APPLICATION_CREDENTIALS` should be **removed**. On the VM the metadata
server supplies credentials, and leaving a stale path set causes authentication
to fail rather than fall back.

There is deliberately no `N8N_USER` or `N8N_PASSWORD` here. Earlier versions of
this file set them to drive the `N8N_BASIC_AUTH_*` variables, which put an HTTP
password prompt in front of the n8n editor. n8n removed that mechanism in
version 1.0 in favour of its own user management, so on the version deployed
here those variables are read and ignored, and keeping them would imply a
protection that is not there. Access is controlled by the owner account created
in Step 8. If you want a password in front of the editor as well, add a
`basicauth` directive to the `Caddyfile` — that runs at the proxy and is still
honoured. An existing deployment may still carry the two variables in its
`.env`; they are harmless and can be left or dropped.

## 7. Start the stack

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f caddy      # watch the certificate being issued
```

The first start takes a few minutes while the image builds and Caddy completes
the ACME challenge. If the certificate fails, the usual cause is that ports 80
and 443 are not open in the firewall, since the HTTP challenge needs port 80.

## 8. Point n8n at the API and Meta at n8n

Open `https://<PUBLIC_HOST>/`. On a fresh volume n8n shows its owner-account
setup screen rather than a password prompt: enter an email and password to
create the account, and use those to sign in from then on. This is n8n's own
user management, not the basic auth the environment variables suggest.

Then import the two workflows. Two changes are needed.

- The HTTP Request node that called the tunnel now calls `http://api:8000/webhook/whatsapp`
- The callback path must match `N8N_CALLBACK_URL`

The inbound workflow (`received-message`) receives from Meta, branches by
message type and posts to the API. The outbound one (`send-reply`) exposes the
`mesra-reply` webhook the worker calls back on and sends the answer to
WhatsApp. Both must be activated, or messages arrive and nothing returns.

Then in the Meta app dashboard, set the WhatsApp webhook callback URL to the n8n
production webhook URL shown in the editor, and verify it.

## 9. Verify end to end

```bash
docker compose exec api curl -fsS http://localhost:8000/health
docker compose logs -f worker
```

Send one message of each modality from WhatsApp — text, a voice note, a photo of
a form and a PDF — and confirm each returns an answer. The worker log shows the
retrieval and generation path for each.

## Operational notes

**Certificates persist** in the `caddy-data` volume. Do not run
`docker compose down -v` casually, since destroying that volume and re-issuing
repeatedly will hit Let's Encrypt rate limits.

**The dead-letter file** is at `/app/data/callback_dlq.jsonl` inside the worker,
on the `worker-data` volume. Inspect it with
`docker compose exec worker cat /app/data/callback_dlq.jsonl` when a reply fails
to reach the user after all retries.

**Ingestion does not run here.** The image excludes Playwright deliberately.
Re-populate the corpus from a developer machine against the same Supabase
project, and the deployed service picks up the new chunks without a redeploy.

**Updating** is `git pull && docker compose up -d --build`. Redis persistence
means in-flight idempotency keys survive the restart.
