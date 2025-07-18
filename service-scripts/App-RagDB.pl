use strict;
use Carp;
use Data::Dumper;
use File::Temp;
use File::Slurp;
use File::Basename;
use IPC::Run 'run';
use JSON;
use Bio::KBase::AppService::AppConfig;
use Bio::KBase::AppService::AppScript;
use Cwd;

our $global_ws;
our $global_token;

#my $script = Bio::KBase::AppService::AppScript->new(\&create_embeddings, \&preflight);
my $script = Bio::KBase::AppService::AppScript->new(\&create_embeddings);
my $rc = $script->run(\@ARGV);
exit $rc;

sub create_embeddings {
    my ($app, $app_def, $raw_params, $params) = @_;
    
    # Get the current directory
    my $cwd = getcwd();

    print "Running embedding app in cwd: $cwd\n";
    
    # Extract parameters from the input
    my $document_folder = $params->{document_folder} || "";
    my $document_file = $params->{document_file} || "";
    my $embedding_endpoint = $params->{embedding_endpoint};
    die "Error: embedding_endpoint parameter is required\n" unless $embedding_endpoint;
    my $api_key = $params->{api_key} || "EMPTY";
    my $model_name = $params->{model_name};
    die "Error: model_name parameter is required\n" unless $model_name;
    my $chunk_size = $params->{chunk_size} || -1;
    my $chunk_overlap = $params->{chunk_overlap} || -1;
    my $output_path = $params->{output_path};
    my $output_file = $params->{output_file};
    my $terminate_on_error = $params->{terminate_on_error} || 0;

    # Construct the command to run the embed_document_corpus.py script
    my @cmd = (
        "embed_document_corpus",
        "--api_key", $api_key,
        "--endpoint", $embedding_endpoint,
        "--model_name", $model_name,
        "--output_folder", $output_path
    );
    
    # Add document_file or document_folder parameter if provided
    if ($document_file) {
        push @cmd, "--document_file", $document_file;
    } elsif ($document_folder) {
        push @cmd, "--document_folder", $document_folder;
    } else {
        die "Error: document_file or document_folder parameter is required\n";
    }
    
    # Add chunk_size and chunk_overlap parameters 
    push @cmd, "--chunk_size", $chunk_size;
    push @cmd, "--chunk_overlap", $chunk_overlap;
    
    # Add terminate_on_error flag
    push @cmd, "--terminate_on_error" if $terminate_on_error;
    
    # Print the command for debugging
    print "Running command: " . join(" ", @cmd) . "\n";
    
    # Run the command
    my $ok = run(\@cmd);
    
    if (!$ok) {
        die "Failed to run embed_document_corpus.py script: $!\n";
    }
    
    # Return the output folder path
    return {
        output_folder => $output_path
    };
}
