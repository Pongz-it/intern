"""Vespa schema deployment script.

Renders Jinja templates and deploys Vespa application package.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from jinja2 import Environment, FileSystemLoader

from agent_rag.core.env_config import vespa_schema_config, vespa_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Template directory relative to this file
TEMPLATE_DIR = Path(__file__).parent.parent / "document_index" / "vespa" / "app_config"


class VespaDeployer:
    """Deploys Vespa application package with rendered schema templates."""

    def __init__(
        self,
        host: str = "localhost",
        config_port: int = 19071,
        query_port: int = 8080,
        timeout: int = 300,
    ):
        """Initialize Vespa deployer.

        Args:
            host: Vespa config server host
            config_port: Vespa config server port (default 19071)
            query_port: Vespa query port (default 8080)
            timeout: Deployment timeout in seconds
        """
        self.host = host
        self.config_port = config_port
        self.query_port = query_port
        self.timeout = timeout
        self.config_url = f"http://{host}:{config_port}"
        self.query_url = f"http://{host}:{query_port}"

    def render_templates(
        self,
        output_dir: Path,
        template_vars: Optional[dict] = None,
    ) -> Path:
        """Render Jinja templates to create Vespa application package.

        Args:
            output_dir: Directory to write rendered files
            template_vars: Template variables (uses env config if not provided)

        Returns:
            Path to the application package directory
        """
        if template_vars is None:
            template_vars = self._get_default_template_vars()

        # Create application package structure
        app_dir = output_dir / "application"
        schemas_dir = app_dir / "schemas"
        schemas_dir.mkdir(parents=True, exist_ok=True)

        # Setup Jinja environment
        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=False,
            keep_trailing_newline=True,
        )

        # Render services.xml
        services_template = env.get_template("services.xml.jinja")
        services_content = services_template.render(**template_vars)
        (app_dir / "services.xml").write_text(services_content)
        logger.info("Rendered services.xml")

        # Render schema file
        schema_template = env.get_template("schemas/agent_rag_chunk.sd.jinja")
        schema_content = schema_template.render(**template_vars)
        schema_name = template_vars.get("schema_name", "agent_rag_chunk")
        (schemas_dir / f"{schema_name}.sd").write_text(schema_content)
        logger.info(f"Rendered schema: {schema_name}.sd")

        # Create validation-overrides.xml to allow destructive changes
        from datetime import datetime, timedelta
        override_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        validation_overrides = f"""<validation-overrides>
    <allow until='{override_date}'>content-cluster-removal</allow>
    <allow until='{override_date}'>schema-removal</allow>
    <allow until='{override_date}'>indexing-change</allow>
    <allow until='{override_date}'>field-type-change</allow>
    <allow until='{override_date}'>field-schema-change</allow>
</validation-overrides>
"""
        (app_dir / "validation-overrides.xml").write_text(validation_overrides)
        logger.info("Created validation-overrides.xml")

        return app_dir

    def _get_default_template_vars(self) -> dict:
        """Get default template variables from environment config."""
        return {
            "schema_name": vespa_schema_config.schema_name,
            "dim": vespa_schema_config.dim,
            "embedding_precision": vespa_schema_config.embedding_precision,
            "multi_tenant": vespa_schema_config.multi_tenant,
            "enable_title_embedding": vespa_schema_config.enable_title_embedding,
            "enable_large_chunks": vespa_schema_config.enable_large_chunks,
            "enable_knowledge_graph": vespa_schema_config.enable_knowledge_graph,
            "enable_access_control": vespa_schema_config.enable_access_control,
            "default_decay_factor": vespa_schema_config.default_decay_factor,
            "rerank_count": vespa_schema_config.rerank_count,
            "redundancy": vespa_schema_config.redundancy,
            "searchable_copies": vespa_schema_config.searchable_copies,
            "search_threads": vespa_schema_config.search_threads,
            "summary_threads": vespa_schema_config.summary_threads,
        }

    def check_health(self) -> bool:
        """Check if Vespa is healthy.

        Returns:
            True if Vespa is responding
        """
        try:
            response = httpx.get(
                f"{self.config_url}/state/v1/health",
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def deploy(
        self,
        app_dir: Path,
        wait_for_convergence: bool = True,
    ) -> bool:
        """Deploy Vespa application package.

        Args:
            app_dir: Path to application package directory
            wait_for_convergence: Whether to wait for deployment convergence

        Returns:
            True if deployment succeeded
        """
        # Create zip archive
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = tmp.name

        try:
            # Create zip file of application package
            # Vespa expects services.xml and schemas/ at root of zip
            shutil.make_archive(
                zip_path.replace(".zip", ""),
                "zip",
                app_dir,  # Root of zip is the app_dir content
                ".",      # Include all files from app_dir
            )

            # Deploy via config server API
            with open(zip_path, "rb") as f:
                response = httpx.post(
                    f"{self.config_url}/application/v2/tenant/default/prepareandactivate",
                    content=f.read(),
                    headers={"Content-Type": "application/zip"},
                    timeout=self.timeout,
                )

            if response.status_code not in (200, 201):
                logger.error(f"Deployment failed: {response.status_code} - {response.text}")
                return False

            logger.info("Application deployed successfully")

            if wait_for_convergence:
                return self._wait_for_convergence()

            return True

        except Exception as e:
            logger.error(f"Deployment error: {e}")
            return False
        finally:
            if os.path.exists(zip_path):
                os.unlink(zip_path)

    def _wait_for_convergence(self, max_retries: int = 60) -> bool:
        """Wait for deployment to converge.

        Args:
            max_retries: Maximum number of retries (1 second each)

        Returns:
            True if converged successfully
        """
        import time

        logger.info("Waiting for deployment convergence...")

        for i in range(max_retries):
            try:
                response = httpx.get(
                    f"{self.config_url}/application/v2/tenant/default/application/default/environment/prod/region/default/instance/default/serviceconverge",
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("converged", False):
                        logger.info("Deployment converged successfully")
                        return True
            except Exception:
                pass

            time.sleep(1)

        logger.warning("Deployment convergence timeout")
        return False

    def validate_deployment(self) -> bool:
        """Validate that deployment is working.

        Returns:
            True if validation succeeded
        """
        try:
            response = httpx.get(
                f"{self.query_url}/state/v1/health",
                timeout=10,
            )
            if response.status_code == 200:
                logger.info("Vespa query endpoint is healthy")
                return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")

        return False


def main():
    """CLI entry point for Vespa deployment."""
    parser = argparse.ArgumentParser(
        description="Deploy Vespa application package for Agent RAG"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("VESPA_HOST", "localhost"),
        help="Vespa host (default: localhost)",
    )
    parser.add_argument(
        "--config-port",
        type=int,
        default=int(os.getenv("VESPA_CONFIG_PORT", "19071")),
        help="Vespa config server port (default: 19071)",
    )
    parser.add_argument(
        "--query-port",
        type=int,
        default=int(os.getenv("VESPA_QUERY_PORT", "8080")),
        help="Vespa query port (default: 8080)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for rendered templates (temp dir if not specified)",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Only render templates, don't deploy",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for deployment convergence",
    )
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Only check Vespa health status",
    )

    args = parser.parse_args()

    deployer = VespaDeployer(
        host=args.host,
        config_port=args.config_port,
        query_port=args.query_port,
    )

    # Health check only
    if args.check_health:
        healthy = deployer.check_health()
        print(f"Vespa health: {'OK' if healthy else 'FAILED'}")
        sys.exit(0 if healthy else 1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="vespa_app_"))
        cleanup = not args.render_only

    try:
        # Render templates
        logger.info(f"Rendering templates to {output_dir}")
        app_dir = deployer.render_templates(output_dir)
        logger.info(f"Application package created at {app_dir}")

        if args.render_only:
            print(f"Templates rendered to: {app_dir}")
            return

        # Check health before deploying
        if not deployer.check_health():
            logger.error("Vespa is not healthy. Please start Vespa first.")
            sys.exit(1)

        # Deploy
        success = deployer.deploy(app_dir, wait_for_convergence=not args.no_wait)

        if success:
            # Validate
            if deployer.validate_deployment():
                logger.info("Vespa deployment completed successfully!")
                sys.exit(0)
            else:
                logger.error("Deployment validation failed")
                sys.exit(1)
        else:
            logger.error("Deployment failed")
            sys.exit(1)

    finally:
        if cleanup and output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
