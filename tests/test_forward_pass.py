from objtracker.train import main


def test_forward_pass(capsys):
    main()
    captured = capsys.readouterr()
    assert "Output shape: torch.Size([4, 1])" in captured.out
