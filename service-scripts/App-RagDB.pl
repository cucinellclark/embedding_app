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

#my $script = Bio::KBase::AppService::AppScript->new(\&create_rag_db, \&preflight);
my $script = Bio::KBase::AppService::AppScript->new(\&create_rag_db);
my $rc = $script->run(\@ARGV);
exit $rc;

sub create_rag_db {
    my ($app, $app_def, $raw_params, $params) = @_;
    
    # Get the current directory
    my $cwd = getcwd();
    
    # Extract parameters from the input
    my $document_folder = $params->{document_folder} || "";
    my $document_file = $params->{document_file} || "";
    my $embedding_endpoint = $params->{embedding_endpoint} || "http://mango.cels.anl.gov:8000/embedding/v1/embeddings";
    my $api_key = $params->{api_key} || "EMPTY";
    my $model_name = $params->{model_name} || "mistralai/Mistral-7B-Instruct-v0.3";
    my $chunk_size = $params->{chunk_size} || -1;
    my $chunk_overlap = $params->{chunk_overlap} || -1;
    my $output_path = $params->{output_path} || "";
    my $output_file = $params->{output_file} || "output.jsonl";
    
    # Construct the full output file path if output_path is provided
    my $full_output_file = $output_file;
    if ($output_path) {
        $full_output_file = "$output_path/$output_file";
    }
    
    # Construct the command to run the embed_document_corpus.py script
    my @cmd = (
        "python", 
        "$cwd/scripts/embed_document_corpus.py",
        "--api_key", $api_key,
        "--endpoint", $embedding_endpoint,
        "--model_name", $model_name,
        "--output_file", $full_output_file
    );
    
    # Add document_file or document_folder parameter if provided
    if ($document_file) {
        push @cmd, "--document_file", $document_file;
    } elsif ($document_folder) {
        push @cmd, "--document_folder", $document_folder;
    }
    
    # Add chunk_size and chunk_overlap parameters if they are not the default values
    if ($chunk_size != -1) {
        push @cmd, "--chunk_size", $chunk_size;
    }
    
    if ($chunk_overlap != -1) {
        push @cmd, "--chunk_overlap", $chunk_overlap;
    }
    
    # Add terminate_on_error flag
    push @cmd, "--terminate_on_error";
    
    # Print the command for debugging
    print "Running command: " . join(" ", @cmd) . "\n";
    
    # Run the command
    my $ok = run(\@cmd);
    
    if (!$ok) {
        die "Failed to run embed_document_corpus.py script: $!\n";
    }
    
    # Return the output file path
    return {
        output_file => $full_output_file
    };
}
