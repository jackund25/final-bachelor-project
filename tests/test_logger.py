from src.utils.logger import setup_logger


def test_setup_logger_writes_to_file(tmp_path):
	log_file = tmp_path / "test.log"
	logger = setup_logger("test_logger_unit", log_file=str(log_file), console=False)

	logger.info("logger-info-message")
	logger.warning("logger-warning-message")
	logger.error("logger-error-message")

	for handler in logger.handlers:
		handler.flush()

	assert log_file.exists()
	content = log_file.read_text(encoding="utf-8")
	assert "logger-info-message" in content
	assert "logger-warning-message" in content
	assert "logger-error-message" in content